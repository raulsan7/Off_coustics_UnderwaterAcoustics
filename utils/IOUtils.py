"""
Module: IOUtils.py
Description: OpenFAST and SubDyn input/output (I/O) parsing utilities.
             Provides optimized readers for text (.out), binary (.outb), 
             and summary (.sum.yaml) simulation results.

             ┌────────────────────────────────────────────────────────┐
             │                      IOUtils.py                        │
             ├────────────────────────────────────────────────────────┤
             │  - get_SDsum_variables ──> Parse node geometries       │
             │  - read_input_SD       ──> Batch structural variables  │
             │  - parse_channels_auto ──> Dual format out/outb router │
             │  - read_curve          ──> 1D/2D linear interpolation  │
             └────────────────────────────────────────────────────────┘

Author: Raul Sanz Ramirez (raul.sanz.ramirez@upm.es / raul.sanz.ramirez@gmail.com)
Institution: Universidad Politecnica de Madrid - ETSIAE
Date: 07/2026 
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d


def get_SDsum_variables(SD_path : str  = None,      # [-] Path to OpenFAST SubDyn sum file
                        Nmembers: int  = 6,         # [-] Number of OpenFAST members
                        Nnodes  : int  = 9,         # [-] Number of OpenFAST nodes  
                        verbose : bool = True):     # [-] Flago to print more info
    """
    Parses SubDyn summary (.SD.sum.yaml) files to reconstruct structural node geometry.

    Mapping Pipeline:
    ┌─────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
    │  SD.sum     │ ──> │ Connection Map:     │ ──> │ Global Node Table:   │
    │  YAML File  │     │ Member -> Node IDs  │     │ ID -> (X, Y, Z)      │
    └─────────────┘     └─────────────────────┘     └──────────────────────┘
                                                               │
                                ┌──────────────────────────────┘
                                ▼
                    ┌──────────────────────┐
                    │ Reshaped Tensor:     │
                    │ (Nmembers, Nnodes, 3)│
                    └──────────────────────┘

    Parameters:
    - SD_path  [str/Path] : Path to the SubDyn summary file.
    - Nmembers [int]      : Number of structural members to process.
    - Nnodes   [int]      : Number of nodes per structural member.
    - verbose  [bool]     : Prints loading status if True.

    Returns:
    - Nodes [ndarray] (Nmembers, Nnodes, 3) [m] : Global initial coordinates of each node.
    """

    if verbose: print(f"Reading SD.sum in {SD_path}")

    # Find members starting line
    line_to_find = f"#Member I Joint1_ID Joint2_ID    Prop_I    Prop_J           Mass         Length     Node IDs..."
    start_line = None
    with open(SD_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line_num, line in enumerate(lines, start=1):
            if line_to_find in line:
                start_line = line_num
                break

    skip_lines = 1
    if start_line is None:
        raise ValueError(f"{line_to_find} not found in file")
    else:
        start_line += skip_lines
        end_line    = start_line + Nmembers - 1

    # Get nodal connections
    j = 0
    connections  = np.zeros((Nmembers,2),dtype=int)
    member_nodes = np.zeros((Nmembers, Nnodes), dtype=int) 
    for i in range(start_line, end_line+1):
        line = lines[i-1].split()
        
        connections[j,0]  = int(line[2])
        connections[j,1]  = int(line[3])

        node_ids = []
        for token in line[8:]:
            if token.startswith('#'):   # if '#' found then is a rigid link (2 nodes only)
                break
            node_ids.append(int(token))

        padded = np.zeros(Nnodes, dtype=int)
        padded[:len(node_ids)] = node_ids
        member_nodes[j,:] = padded
        j += 1

    # Find nodes starting line
    line_to_find = f"#     Node_[#]          X_[m]           Y_[m]           Z_[m]"
    start_line = None
    with open(SD_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line_num, line in enumerate(lines, start=1):
            if line_to_find in line:
                start_line = line_num
                break

    skip_lines = 2
    number_of_nodes = int(lines[start_line].split()[2])
    if start_line is None:
        raise ValueError(f"{line_to_find} not found in file")
    else:
        start_line += skip_lines
        end_line = start_line + number_of_nodes - 1

    # Placed as a list, first the Joints, then the interior points
    Nodes_flat = np.zeros((number_of_nodes,3))
    Nodes = np.zeros((Nmembers,Nnodes,3))


    for i in range(start_line, end_line+1):
        line = lines[i-1].split()
            
        # Clear the comma that comes with data
        idx = int(float(line[2].rstrip(',')))
        x   = float(line[3].rstrip(','))
        y   = float(line[4].rstrip(','))
        z   = float(line[5].rstrip(','))

        Nodes_flat[idx-1,:] = np.array([x,y,z], dtype=float)

    for i in range(Nmembers):
        for j in range(Nnodes):
            nid = member_nodes[i, j]
            if nid > 0:
                Nodes[i, j, :] = Nodes_flat[nid - 1, :]
            else:
                Nodes[i, j, :] = np.nan
    
    return Nodes

def read_input_SD(filename: str = None,             # [-] Path to OpenFAST SubDyn sum file
                  what    : str = "acceleration",   # [-] Name of the output to read
                  skip    : int = 1,                # [-] Skips time data e.g. Time[::skip]
                  Nmembers: int = 8,                # [-] Number of OpenFAST members 
                  Nnodes  : int = 5,                # [-] Number of OpenFAST nodes   
                  From    : int = 0.0,              # [-] Starts reading from here (0. <= From < 1.) 
                  Upto    : int = 1.0,              # [-] Ends up reading     here (0. < Upto <= 1.) 
                  verbose : int = False):           # [-] Flag to print more info
    """
    Batches structural variables (accelerations, forces, etc.) from SubDyn output channels.

    Automatic Channel Naming Scheme:
    ┌────────────────────────────────────────────────────────┐
    │ Channel = M{member_id}N{node_id}{variable_suffix}      │
    │ Example (Accel. X): M1N1TAxe                           │
    └────────────────────────────────────────────────────────┘

    Parameters:
    - filename [str/Path]  : Path to the OpenFAST output file.
    - what     [str]       : Output type ('displacement', 'acceleration', 'force', 
                             'momentum', 'int_displacement', 'int_acceleration').
    - skip     [int]       : Decimation factor for time steps (downsampling).
    - Nmembers [int]       : Total structural members in the simulation.
    - Nnodes   [int]       : Structural nodes per member.
    - From     [float]     : Time window start index fraction [0.0 - 1.0].
    - Upto     [float]     : Time window end index fraction [0.0 - 1.0].
    - verbose  [bool]      : Prints extracted channel naming information if True.

    Returns:
    - Time  [ndarray] (nt,) [s]                        : Downsampled time array.
    - array [ndarray] (nt, Nmembers, Nnodes, 3) [unit] : Reshaped time-series tensor.
    - units [str]                                      : Physical units of the retrieved variable.
    """

    # Only one output for subdyn
    if np.size(filename) > 1: filename = filename[0]

    # Case selection
    if what == "displacement":
        strx = "TDxss"; stry = "TDyss"; strz = "TDzss"
    elif what == "acceleration":
        strx = "TAxe"; stry = "TAye"; strz = "TAze"
    elif what == "force":
        strx = "FKxe"; stry = "FKye"; strz = "FKze"
    elif what == "momentum":
        strx = "MKxe"; stry = "MKye"; strz = "MKze"
    elif what == "int_displacement":
        strx = "IntfTDXss"; stry = "IntfTDYss"; strz = "IntfTDZss"
    elif what == "int_acceleration":
        strx = "IntfTAXss"; stry = "IntfTAYss"; strz = "IntfTAZss"
    else:
        raise ValueError(f"Invalid option in 'what = {what}' variable \n Input 'displacement', 'acceleration', 'force', 'momentum', 'int_displacement' or 'int_acceleration'")
    
    if verbose: print(f"Reading {filename} --> {what}: [MiNj{strx}, MiNj{stry}, MiNj{strz}]")

    M, N = "M", "N"
    outputchannels = []     # All output channel list

    for i in range(1, Nmembers+1):
        for j in range(1,Nnodes+1):
            outputchannels += [M+str(i)+N+str(j)+strx, M+str(i)+N+str(j)+stry, M+str(i)+N+str(j)+strz]
    Time, array, units = parse_channels_auto(filename, plotChannels=outputchannels, From=From, Upto=Upto, verbose=verbose)
    nt = Time.size

    array = array.reshape((nt, Nmembers, Nnodes, 3))
    array = array[::skip]
    Time = Time[::skip]
    units = units[0]

    return Time, array, units

def read_input_AD(filename: str = None,             # [-] Path to OpenFAST output file
                  what    : str = "Fn",             # [-] Name of the output to read
                  skip    : int = 1,                # [-] Skips time data e.g. Time[::skip]
                  Nnodes  : int = None,             # [-] Number of nodes in the blade
                  Blade   : int = 1,                # [-] Number of blade to output
                  From    : int = 0.0,              # [-] Starts reading from here (0. <= From < 1.) 
                  Upto    : int = 1.0,              # [-] Ends up reading     here (0. < Upto <= 1.) 
                  verbose : int = False):           # [-] Flag to print more info
    """
    Reads all nodes outputs from .out/.outb
    """

    if Nnodes <= 0:
        raise ValueError("IOUtils.read_input_AD(): Nnodes must be a positive integer")

    string = f"AB{Blade}N"
    outputchannels = []

    for i in range(Nnodes):
        outputchannels.append(f"{string}{i + 1:03d}"+what)

    Time, array, units = parse_channels_auto(filename, outputchannels, From=From, Upto=Upto, verbose=verbose)

    return Time, array, units

def parse_channels_auto(full_path         : str = None,     # [-] Path to OpenFAST output file
                        plotChannels      : list = None,    # [-] List of OpenFAST channels to output
                        From              : float = 0.0,    # [-] Starts reading from here (0. <= From < 1.) 
                        Upto              : float = 1.0,    # [-] Ends up reading     here (0. < Upto <= 1.) 
                        available_channels: bool = False,   # [-] Available channels
                        verbose           : bool = True,    # [-] Flag to print more info
                        chunk_size        : int = 4096):    # [-] Number of rows to decode per chunk
    """
    Detects OpenFAST output file extensions and routes parsing to the correct reader.

    File Routing:
    ┌───────────────────────────────┐
    │       OpenFAST File Path      │
    └───────────────┬───────────────┘
                    │
            ┌───────┴───────┐
            │   Extension?  │
            └─┬───────────┬─┘
              ▼ (.outb)   ▼ (.out)
      ┌───────────────┐   ┌───────────────┐
      │  Binary Parser│   │  ASCII Parser │
      └───────────────┘   └───────────────┘

    Parameters:
    - full_path          [str/Path] : OpenFAST output file path.
    - plotChannels       [list]     : List of target channel names to extract.
    - From               [float]    : Start window fraction [0.0 - 1.0].
    - Upto               [float]    : End window fraction [0.0 - 1.0].
    - available_channels [bool]     : Prints all available channel headers if True.
    - verbose            [bool]     : Verbose prints showing active routing decision.

    Returns:
    - time  [ndarray] (nt,) [s]         : Time-series steps.
    - array [ndarray] (nt, Nchannels)   : Extracted timeseries columns.
    - units [list of str]               : Units list corresponding to plotChannels.
    """

    full_path = Path(full_path)
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")

    ext = full_path.suffix.lower()

    if ext == ".outb":
        if verbose:
            print(f"Detected binary OpenFAST output (.outb): {full_path.name}")
        return parse_channels_binary(
            full_path=full_path,
            plotChannels=plotChannels,
            From=From,
            Upto=Upto,
            available_channels=available_channels,
            verbose=verbose,
            chunk_size=chunk_size,
        )

    elif ext == ".out":
        if verbose:
            print(f"Detected text OpenFAST output (.out): {full_path.name}")
        return parse_channels(
            full_path=full_path,
            plotChannels=plotChannels,
            From=From,
            Upto=Upto,
            available_channels=available_channels,
            verbose=verbose
        )

    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Expected .out or .outb."
        )

def parse_channels(full_path         : str = None,     # [-] Path to OpenFAST output file
                   plotChannels      : list = None,    # [-] List of OpenFAST channels to output
                   From              : float = 0.0,    # [-] Starts reading from here (0. <= From < 1.) 
                   Upto              : float = 1.0,    # [-] Ends up reading     here (0. < Upto <= 1.) 
                   available_channels: bool = False,   # [-] Available channels
                   verbose           : bool = True):   # [-] Flag to print more info
    """
    Parses OpenFAST ASCII text files (.out).

    Structure of .out files:
    ┌────────────────────────────────────────────────────────┐
    │ Line 1-6 : Header description metadata                 │
    │ Line 7   : Channel Names (e.g. Time, WindVxi, ...)     │
    │ Line 8   : Channel Units (e.g. s, m/s, ...)            │
    │ Line 9+  : Numerical values                              │
    └────────────────────────────────────────────────────────┘

    Parameters:
    - full_path          [str/Path] : OpenFAST .out text file path.
    - plotChannels       [list]     : Target channel headers to parse.
    - From               [float]    : Start index window fraction [0.0 - 1.0].
    - Upto               [float]    : End index window fraction [0.0 - 1.0].
    - available_channels [bool]     : If True, prints all file channel headers.
    - verbose            [bool]     : Displays status info on completed reading.

    Returns:
    - time  [ndarray] (nt,) [s]       : Extracted simulation time.
    - array [ndarray] (nt, Nchannels) : Signal arrays matching plotChannels.
    - units [list of str]             : Metadata units of extracted columns.
    """

    full_path = Path(full_path)
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")
    
    if not (0 <= From <= Upto <= 1.0):
        raise ValueError("Fractions must satisfy: 0 <= From <= Upto <= 1.0")
    
    if not plotChannels:
        raise ValueError("plotChannels cannot be empty")

    n_chan = len(plotChannels)

    with open(full_path) as fid:
        headers = [fid.readline() for _ in range(6)]    # Read Header
        channels = fid.readline().strip().split()       # Channel names
        uunits = fid.readline().strip().split()         # Channel units
    
    data = np.loadtxt(full_path, skiprows=8)            # Time series
    if available_channels: print('Available Channels: ',channels)

    # Apply time range filtering
    N_total = data.shape[0]
    start_idx = int(N_total * From)
    end_idx = int(N_total * Upto)
    
    data = data[start_idx:end_idx, :]
    time = data[:, 0]
    N_filtered = data.shape[0]

    # Create output array with the filtered data
    array = np.zeros((N_filtered, n_chan))
    units = []
    for i in range(n_chan):
        idx = channels.index(plotChannels[i])
        array[:, i] = data[:, idx]
        units.append(uunits[idx])
    
    if verbose:
        print(f"Output array shape is {array.shape}")
        print(f"Output Channels: {plotChannels}")
        print(f"Units: {units}")

    return time, array, units

def parse_channels_binary(full_path         : str = None,     # [-] Path to OpenFAST output file
                          plotChannels      : list = None,    # [-] List of OpenFAST channels to output
                          From              : float = 0.0,    # [-] Starts reading from here (0. <= From < 1.) 
                          Upto              : float = 1.0,    # [-] Ends up reading     here (0. < Upto <= 1.) 
                          available_channels: bool = False,   # [-] Available channels
                          verbose           : bool = True,    # [-] Flag to print more info
                          chunk_size        : int = 4096):    # [-] Rows decoded per chunk
    """
    Parses OpenFAST compressed binary output files (.outb).

    The parser now streams the file in chunks so it does not allocate the full
    time/channel matrix in memory. This avoids the out-of-memory condition that
    can happen for large OpenFAST outputs.

    Parameters:
    - full_path          [str/Path] : OpenFAST .outb binary file path.
    - plotChannels       [list]     : Target channel headers to parse.
    - From               [float]    : Start window fraction [0.0 - 1.0].
    - Upto               [float]    : End window fraction [0.0 - 1.0].
    - available_channels [bool]     : Prints all decoded file channel headers if True.
    - verbose            [bool]     : Prints binary metadata description if True.
    - chunk_size         [int]      : Number of rows to decode at a time.

    Returns:
    - time  [ndarray] (nt,) [s]       : Extracted simulation time.
    - array [ndarray] (nt, Nchannels) : De-scaled data arrays matching plotChannels.
    - units [list of str]             : Physical units for each target channel.
    """

    # --- Checks
    full_path = Path(full_path)
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")
    if not (0 <= From <= Upto <= 1.0):
        raise ValueError("Fractions must satisfy: 0 <= From <= Upto <= 1.0")
    if not plotChannels:
        raise ValueError("plotChannels cannot be empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    # --- Open file
    with open(full_path, "rb") as fid:
        # --- Read header
        FileID = np.fromfile(fid, np.int16, 1)[0]

        # File format constants
        FileFmtID = {
            "WithTime": 1,
            "WithoutTime": 2,
            "NoCompressWithoutTime": 3,
            "ChanLen_In": 4,
        }

        if FileID == FileFmtID["ChanLen_In"]:
            LenName = np.fromfile(fid, np.int16, 1)[0]
        else:
            LenName = 10

        NumOutChans = np.fromfile(fid, np.int32, 1)[0]
        NT = np.fromfile(fid, np.int32, 1)[0]

        # --- Time info
        if FileID == FileFmtID["WithTime"]:
            TimeScl = np.fromfile(fid, np.float64, 1)[0]
            TimeOff = np.fromfile(fid, np.float64, 1)[0]
        else:
            TimeOut1 = np.fromfile(fid, np.float64, 1)[0]
            TimeIncr = np.fromfile(fid, np.float64, 1)[0]

        # --- Channel scaling
        if FileID == FileFmtID["NoCompressWithoutTime"]:
            ColScl = np.ones(NumOutChans, dtype=np.float32)
            ColOff = np.zeros(NumOutChans, dtype=np.float32)
        else:
            ColScl = np.fromfile(fid, np.float32, NumOutChans)
            ColOff = np.fromfile(fid, np.float32, NumOutChans)

        # --- Description string
        LenDesc = np.fromfile(fid, np.int32, 1)[0]
        DescStr = fid.read(LenDesc).decode("ascii", errors="ignore")

        # --- Channel names
        ChanName = []
        for _ in range(NumOutChans + 1):
            raw = fid.read(LenName)
            ChanName.append(raw.decode("ascii", errors="ignore").strip())

        # --- Channel units
        ChanUnit = []
        for _ in range(NumOutChans + 1):
            raw = fid.read(LenName)
            ChanUnit.append(raw.decode("ascii", errors="ignore").strip())

        if available_channels:
            print("Available Channels:", ChanName)

        start_idx = int(NT * From)
        end_idx = int(NT * Upto)
        if start_idx < 0:
            start_idx = 0
        if end_idx > NT:
            end_idx = NT
        n_target_rows = max(0, end_idx - start_idx)

        time_out = np.empty(n_target_rows, dtype=np.float64)
        array_out = np.empty((n_target_rows, len(plotChannels)), dtype=np.float64)

        # --- Stream data in chunks
        written = 0
        row_counter = 0
        while row_counter < end_idx:
            rows_this_chunk = min(chunk_size, end_idx - row_counter)

            if FileID == FileFmtID["WithTime"]:
                packed_time_chunk = np.fromfile(fid, np.int32, rows_this_chunk)
            else:
                packed_time_chunk = None

            if FileID == FileFmtID["NoCompressWithoutTime"]:
                packed_data_chunk = np.fromfile(fid, np.float64, rows_this_chunk * NumOutChans)
            else:
                packed_data_chunk = np.fromfile(fid, np.int16, rows_this_chunk * NumOutChans)

            if packed_data_chunk.size == 0:
                break

            n_rows = min(rows_this_chunk, packed_data_chunk.size // NumOutChans)
            if packed_time_chunk is not None:
                n_rows = min(n_rows, packed_time_chunk.size)

            if n_rows <= 0:
                break

            data_chunk = packed_data_chunk[:n_rows * NumOutChans].reshape((n_rows, NumOutChans))
            if FileID == FileFmtID["WithTime"]:
                time_chunk = (packed_time_chunk[:n_rows] - TimeOff) / TimeScl
            else:
                time_chunk = TimeOut1 + TimeIncr * np.arange(row_counter, row_counter + n_rows)

            if row_counter + n_rows <= start_idx:
                row_counter += n_rows
                continue

            slice_start = max(0, start_idx - row_counter)
            slice_end = min(n_rows, end_idx - row_counter)
            if slice_end <= slice_start:
                row_counter += n_rows
                continue

            data_chunk = data_chunk[slice_start:slice_end]
            time_chunk = time_chunk[slice_start:slice_end]

            if written + len(time_chunk) > n_target_rows:
                time_chunk = time_chunk[: n_target_rows - written]
                data_chunk = data_chunk[: n_target_rows - written]

            if len(time_chunk) == 0:
                break

            data_chunk = (data_chunk - ColOff) / ColScl

            for i, ch in enumerate(plotChannels):
                if ch not in ChanName:
                    raise ValueError(f"Channel '{ch}' not found. Available: {ChanName}")
                idx = ChanName.index(ch) - 1
                array_out[written:written + len(time_chunk), i] = data_chunk[:, idx]

            time_out[written:written + len(time_chunk)] = time_chunk
            written += len(time_chunk)
            row_counter += n_rows

            if written >= n_target_rows:
                break

    if n_target_rows == 0:
        return np.array([], dtype=np.float64), np.empty((0, len(plotChannels)), dtype=np.float64), []

    # --- Extract requested channels
    units = []
    for ch in plotChannels:
        if ch not in ChanName:
            raise ValueError(f"Channel '{ch}' not found. Available: {ChanName}")
        idx = ChanName.index(ch)
        units.append(ChanUnit[idx])

    if verbose:
        print(f"Output array shape is {array_out[:written].shape}")
        print(f"Output Channels: {plotChannels}")
        print(f"Units: {units}")
        print(f"Description: {DescStr}")

    return time_out[:written], array_out[:written], units

def read_curve(filename: str = None,    # [-] Path to de file
               cols    : int = 2):      # [-] Number of columns to read
    """
    Parses multi-column CSV datasets to construct linear interpolators.

    Typical Use-case:
    ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
    │ Input CSV Curve  │ ──> │ numpy.genfromtxt │ ──> │ interp1d Object  │
    │ (e.g. RPM vs WS) │     │                  │     │   (Extrapolate)  │
    └──────────────────┘     └──────────────────┘     └──────────────────┘

    Parameters:
    - filename [str/Path] : Target CSV curve file. First column acts as independent variable.
    - cols     [int]      : Number of columns to expect.

    Returns:
    - interpolator [scipy.interpolate.interp1d or ndarray] : 
        - If cols == 2: Returns a single 1D interpolation function.
        - If cols > 2: Returns an array of independent interpolation functions.

    Raises:
    - ValueError: If cols < 2 (requires at least one independent and one dependent variable).
    """
    
    if cols == 2:
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        ws_array = data[:,0]
        var = data[:,1]
        return interp1d(ws_array, var, kind='linear', fill_value='extrapolate')
    elif cols > 2:
        output = []
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        ws_array = data[:,0]
        
        for i in range(1, cols):
            var = data[:, i]
            interp_func = interp1d(ws_array, var, kind='linear', fill_value='extrapolate')
            output.append(interp_func)
        
        return np.array(output)
    else:
        raise ValueError("Number of columns must be: cols >= 2")



