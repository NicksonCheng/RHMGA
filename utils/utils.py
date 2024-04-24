import yaml
import os


def load_config(args, path):
    with open(path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    config = config[args.dataset]

    for k, v in config.items():
        setattr(args, k, v)
    return args


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
    curr_file = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.join(curr_file, "..")
    if file == "log":
        log_path = os.path.join(project_path, "log", "performance")
        print(log_path)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        file_name = f"./{file}/performance/{log_times}_HGARME("
    else:
        img_path = os.path.join(project_path, "img")
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        file_name = f"./{file}/{log_times}_HGARME("
    if args.edge_recons:
        file_name += "edge"
        if args.all_edge_recons:
            file_name += "[all]"
        file_name += "_"
    if args.feat_recons:
        file_name += "feat"
        if args.all_feat_recons:
            file_name += "[all]"

    file_name += f")_{args.dataset}"
    if file == "log":
        file_name += ".txt"
    else:
        file_name += ".png"
    return file_name
