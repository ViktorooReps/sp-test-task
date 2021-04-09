import matplotlib.pyplot as plt
from utils.plotter import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=int, default=0)
    parser.add_argument("--loggers", action="store_true")

    args = parser.parse_args()

    if args.loggers:
        logger = load_obj("i500s100/active_logger")
        rand_logger = load_obj("i500s100/active_logger_rand")
        plot_from_active_logger(logger, rand_logger, name="active_comp_i500_s100")
    else:
        args = parser.parse_args()

        plot_last_run()
        plot_in_comparison(args.since)