def colorize(string, color):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    return f"{colors[color]}{string}{colors['reset']}"


def name_file(args, file, log_times):
    if file == "log":
        file_name = f"./{file}/performance/HGARME("
    else:
        file_name = f"./{file}/HGARME("
    print(args)
    if args.edge_recons:
        file_name += "edge"
        if args.all_edge_recons:
            file_name += "[all]"
        file_name += "_"
    if args.feat_recons:
        file_name += "feat"
        if args.all_feat_recons:
            file_name += "[all]"

    file_name += f")_{args.dataset}_{log_times}"
    if file == "log":
        file_name += ".txt"
    else:
        file_name += ".png"
    print(file_name)
    return file_name
