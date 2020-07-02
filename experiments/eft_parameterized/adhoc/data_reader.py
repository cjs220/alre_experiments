import configparser, pickle, os, sys
import numpy as np
from matplotlib import pyplot as plt


#  Brief: function turns a string into a vector, assuming string formatted as "['A', 'B', 'C']" or "[A, B, C]"
#  - simple non-generalisable function, only works assuming the correct formatting
#
def str_to_vec(string, delim=","):
    string = string.replace("[", "")
    string = string.replace("]", "")
    string = string.replace(" ", "")
    string = string.replace("'", "")
    return [x for x in string.split(delim) if len(x) > 0]


#  Class DataTable
#    -  store event-level observable values and weights, observable names and their types
#    -  can be saved to pickle file
#    -  can be loaded from pickle file, or from a text file produced by Rivet
#
class DataTable:
    #
    #  Brief:  Constructor
    #     - If fname provided, we will load it if we recognise the extension as a text- or pickle-file
    #
    def __init__(self, fname=None):
        self.keys = None
        self.types = None
        self.weights = None
        self.data = None
        self.xsec_per_event = None
        if type(fname) == type(None): return
        fname_extension = fname.split(".")[-1]
        if fname_extension in ["dat", "text", "txt"]:
            self.load_from_text_file(fname)
        elif fname_extension in ["pickle", "p"]:
            self.load_from_pickle_file(fname)
        else:
            raise RuntimeWarning(
                f"File type extension not recognised from filename {fname}. Returning without loading.")

    #
    #  Brief:  Combine two data tables
    #
    def __add__(self, other):
        assert self.keys == other.keys
        assert self.types == other.types
        assert [w for w in self.weights] == [w for w in other.weights]
        ret = DataTable()
        ret.keys = self.keys
        ret.types = self.types
        ret.data = np.concatenate([self.data, other.data])
        ret.weights = {}
        n1, n2 = self.xsec_per_event, other.xsec_per_event
        ret.xsec_per_event = 1.
        for wname, self_weights in self.weights.items():
            other_weights = other.weights[wname]
            ret.weights[wname] = np.concatenate([n1 * self_weights, n2 * other_weights])
        return ret

    #
    #  Brief:  Filter values
    #    - remove rows which fail the filter condition
    #
    def filter(self, key, minimum=None, maximum=None, remove=False):
        do_min = False if type(minimum) == type(None) else True
        do_max = False if type(maximum) == type(None) else True
        if (not do_min) and (not do_max):
            raise RuntimeError("must provide a minimum and/or maximum value for the filter")
        if (type(self.data) == type(None)) or (self.get_num_events() < 1):
            raise RuntimeError(f"No data found")
        col_idx = self.get_column_index(key)
        new_data, surviving_indices = [], []
        for row_idx, row in enumerate(self.data):
            val = row[col_idx]
            if (do_min and val < minimum) or (do_max and val > maximum):
                continue
            new_data.append(row)
            surviving_indices.append(row_idx)
        if len(new_data) < len(self.data):
            for weight, weights in self.weights.items():
                self.weights[weight] = weights[surviving_indices]
        self.data = np.array(new_data)
        if remove:
            self.remove_column(col_idx)

    #
    #  Brief:  Return the index of the column of data corresponding to a given key
    #
    def get_column_index(self, key):
        self._assert_key_exists(key)
        return self.keys.index(key)

    #
    #  Brief:  Return the number of stored events
    #
    def get_num_events(self):
        if type(self.data) == type(None):
            return -1
        return self.data.shape[0]

    #
    #  Brief:  Return the number of stored keys
    #
    def get_num_observables(self):
        if type(self.keys) == type(None):
            return -1
        return len(self.keys)

    #
    #  Brief:  Return the number of stored weights
    #
    def get_num_weights(self):
        if type(self.weights) == type(None):
            return -1
        return len(self.weights)

    #
    #  Brief:  Return the observables and weights as separate arrays
    #
    def get_observables_and_weights(self, weight=None):
        return self.data, self.get_weights(weight)

    #
    #  Brief:  Return the types of all observables, in order, ignoring any key called "weight"
    #
    def get_observables_and_types(self):
        return [(k, t) for t, k in zip(self.types, self.keys)]

    #
    #  Brief:  Return the number of stored events
    #
    def get_sum_weights(self, weight=None):
        return np.sum(self.get_weights(weight))

    #
    #  Brief:  Return the sum of squared weights
    #
    def get_sum_squared_weights(self, weight=None):
        weights = self.get_weights(weight)
        return np.sum(weights * weights)

    #
    #  Brief:  Return the total cross section of all events
    #
    def get_total_xsec(self, weight=None):
        if type(self.xsec_per_event) is None: return None
        if type(self.weights) is None: return None
        return self.get_sum_weights(weight) * self.xsec_per_event

    #
    #  Brief:  Return the total cross section of all events, with its Poisson error
    #
    def get_total_xsec_and_error(self, weight=None):
        if type(self.xsec_per_event) is None: return None
        if type(self.weights) is None: return None
        return self.get_sum_weights(weight) * self.xsec_per_event, np.sqrt(
            self.get_sum_squared_weights(weight)) * self.xsec_per_event

    #
    #  Brief:  Return the weights as a 1D array
    #          - if no weights stored, assume uniform weights which sum to 1
    #
    def get_weights(self, weight=None):
        if type(self.weights) == type(None):
            return None
        if type(weight) == type(None):
            if self.get_num_weights() == 0:
                num_events = self.get_num_events()
                return np.full(fill_value=1. / num_events, shape=(num_events,))
            if self.get_num_weights() == 1:
                weight = [w for w in self.weights][0]
                return self.get_weights(weight)
        if weight not in self.weights:
            raise KeyError(f"Weight '{weight}' not found")
        return self.weights[weight]

    #
    #  Brief:  Load values from a pickle file
    #    - Pickle file created by this class (see self.save_to_pickle_file())
    #
    def load_from_pickle_file(self, fname):
        if not os.path.isfile: raise RuntimeError(f"File {fname} does not exist")
        #
        #   Load pickle file, and make sure the expected entries are present
        #
        f = pickle.load(open(fname, "rb"))
        if type(f) != dict: raise TypeError(
            f"Expected contents of pickle file {fname} to be type {dict} but {type(f)} found")
        for required_entry in ["Keys", "Types", "Data", "Weights", "Xsec_per_event"]:
            if required_entry in f: continue
            raise RuntimeError(f"Entry {required_entry} not found in pickle file {fname}")
        #
        #   Save values
        #
        self.keys = f["Keys"]
        self.types = f["Types"]
        self.data = f["Data"]
        self.weights = f["Weights"]
        self.xsec_per_event = f["Xsec_per_event"]

    #
    #  Brief:  Load values from a text file
    #    - Text file is produce by a Rivet routine, and must be formatted correctly
    #
    def load_from_text_file(self, fname):
        if not os.path.isfile: raise RuntimeError(f"File {fname} does not exist")
        #
        #   Load config file, and make sure the expected entries are present
        #
        config = configparser.ConfigParser()
        config.read(fname)
        if "DATA" not in config.sections():
            raise RuntimeError(f"Header DATA not found in file {fname}")
        data = config["DATA"]
        for required_entry in ["Keys", "Events", "Xsec_per_event"]:
            if required_entry in data: continue
            raise RuntimeError(f"Entry [DATA]::{required_entry} not found in file {fname}")
        #
        #   Load the keys from the config file
        #
        self.keys, self.types = [], []
        for entry in str_to_vec(data["Keys"]):
            split_entry = entry.split("::")
            if len(split_entry) == 2:
                str_type = split_entry[0]
                if str_type.lower() == "float":
                    self.types.append(float)
                elif str_type.lower() == "int":
                    self.types.append(int)
                else:
                    raise RuntimeError(f"Key {entry} type not recognised")
                self.keys.append(split_entry[1])
            elif len(split_entry) == 1:
                self.types.append(float)
                self.keys.append(split_entry[0])
            else:
                raise RuntimeError(f"Key {entry} format not recognised")
        num_keys = len(self.keys)
        #
        #   Load the event data from the config file
        #
        events = []
        for line in data["Events"].split("\n"):
            vec_line = str_to_vec(line)
            if len(vec_line) != num_keys:
                raise RimetimeError(f"Event '{line}' has {len(vec_line)} entries where {num_keys} expected")
            row = []
            for entry, typ in zip(vec_line, self.types):
                row.append(typ(entry))
            events.append(row)
        self.data = np.array(events)
        self.xsec_per_event = float(data["Xsec_per_event"])
        self.weights = {}
        if "weight" in self.keys:
            self.set_key_as_weight("weight")

    #
    #  Brief:  Plot contents
    #
    def plot_contents(self, show=True, save=None):
        keys, data = self.keys, self.data
        num_keys = self.get_num_observables()
        weights = self.get_weights() * self.xsec_per_event
        num_columns = 6
        num_rows = np.ceil(num_keys / 6)
        fig = plt.figure(figsize=(2 * num_columns, 2 * num_rows))
        for key_idx in range(num_keys):
            ax = fig.add_subplot(num_rows, num_columns, 1 + key_idx)
            data_x = np.array([x for x in data[:, key_idx] if x != -99])
            data_w = np.array([w for x, w in zip(data[:, key_idx], weights) if x != -99])
            entries, bins, patches = ax.hist(data_x, weights=data_w)
            ax.set_title(f"{keys[key_idx]}", weight="bold", fontsize=15)
            if (np.min(entries) >= 0):
                ax.set_yscale("log")
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.4, hspace=0.4)
        if type(save) != type(None):
            plt.savefig(save, bbox_inches="tight")
        if show:
            plt.show()

    #
    #  Brief:  Print the loaded keys and data summary
    #
    def print_summary(self):
        key_width, type_width = max(4, max([len(s) for s in self.keys])), max(4, max([len(str(t)) for t in self.types]))
        header = f"    | Column | {'Name'.ljust(key_width)} | {'Type'.ljust(type_width)} |"
        line_width = len(header) - 4
        num_keys = len(self.keys)
        print(f"* Registered the following keys:")
        print("    +-" + "-" * (line_width - 4) + "-+")
        print(f"{header}")
        print("    +-" + "-" * (line_width - 4) + "-+")
        for col_idx, (key, typ) in enumerate(zip(self.keys, self.types)):
            print(f"    | {str(col_idx).ljust(6)} | {key.ljust(key_width)} | {str(typ).ljust(type_width)} |")
        print("    +-" + "-" * (line_width - 4) + "-+")

        key_width = max(4, max([len(w) for w in self.weights]))
        header = f"    | {'Name'.ljust(key_width)} | Sum of weights |"
        line_width = len(header) - 4
        print(f"\n* Registered the following weights:")
        print("    +-" + "-" * (line_width - 4) + "-+")
        print(f"{header}")
        print("    +-" + "-" * (line_width - 4) + "-+")
        for name in self.weights:
            str_sum_of_weights = f"{self.get_sum_weights(name):.4f}"
            print(f"    | {name.ljust(key_width)} | {str_sum_of_weights.ljust(14)} |")
        print("    +-" + "-" * (line_width - 4) + "-+")

        print(f"\n* Number of events       : {self.get_num_events()}")
        print(f"* Cross section per event: {self.xsec_per_event}")

        if self.get_num_weights() == 1:
            total_xsec, total_xsec_err = self.get_total_xsec_and_error()
            print(f"* Total cross section    : {total_xsec} +/- {total_xsec_err} pb")

    #
    #  Brief:  Reorder the data table columns
    #
    def reorder(self, *keys):
        num_keys = self.get_num_observables()
        if len(keys) != num_keys:
            raise RuntimeError(f"{len(keys)} keys provided where {num_keys} expected")
        for key in keys:
            self._assert_key_exists(key)
        ordered_indices = [self.get_column_index(key) for key in keys]
        self.keys = [self.keys[i] for i in ordered_indices]
        self.types = [self.types[i] for i in ordered_indices]
        self.data = self.data[:, ordered_indices]

    #
    #  Brief:  Remove the specified column from the data table
    #
    def remove_column(self, key):
        key_idx = self.get_column_index(key)
        self.data = np.delete(self.data, key_idx, axis=1)
        del self.keys[key_idx]
        del self.types[key_idx]

    #
    #  Brief:  Remove the specified weight from the list of weights
    #
    def remove_weight(self, weight):
        if weight not in self.weights:
            raise KeyError(f"Weight '{weight}' does not exist")
        del self.weights[weight]

    #
    #  Brief:  save the table contents to a pickle file (in a format which can be loaded elsewhere)
    #
    def save_to_pickle_file(self, fname):
        if type(self.keys) == type(None): raise RuntimeError("Cannot save data whilst self.keys is None")
        if type(self.types) == type(None): raise RuntimeError("Cannot save data whilst self.types is None")
        if type(self.data) == type(None): raise RuntimeError("Cannot save data whilst self.data is None")
        if type(self.weights) == type(None): raise RuntimeError("Cannot save data whilst self.weights is None")
        if type(self.xsec_per_event) == type(None): raise RuntimeError(
            "Cannot save data whilst self.xsec_per_event is None")
        to_save = {}
        to_save["Keys"] = self.keys
        to_save["Types"] = self.types
        to_save["Data"] = self.data
        to_save["Weights"] = self.weights
        to_save["Xsec_per_event"] = self.xsec_per_event
        pickle.dump(to_save, open(fname, "wb"))

    #
    #  Brief:  Scale the cross section by a given factor
    #
    def scale(self, factor):
        self.xsec_per_event = factor * self.xsec_per_event

    #
    #  Brief:  save the table contents to a pickle file (in a format which can be loaded elsewhere)
    #
    def set_key_as_weight(self, key):
        if key not in self.keys:
            raise KeyError(f"Key '{key}' not defined")
        key_idx = self.get_column_index(key)
        self.weights[key] = self.data[:, key_idx].copy()
        self.remove_column(key)

    #
    #  Brief:  Swap two data columns
    #
    def swap_columns(self, key1, key2):
        col_idx1 = self.get_column_index(key1)
        col_idx2 = self.get_column_index(key2)
        self.data[:, col_idx1], self.data[:, col_idx2] = self.data[:, col_idx2], self.data[:, col_idx1].copy()
        self.keys[col_idx1], self.keys[col_idx2] = self.keys[col_idx2], self.keys[col_idx1]
        self.types[col_idx1], self.types[col_idx2] = self.types[col_idx2], self.types[col_idx1]

    #
    #  Brief:  Throw an error if the given key is not defined in self.keys
    #
    def _assert_key_exists(self, key):
        if type(self.keys) == type(None):
            raise RuntimeError(f"Keys have not been set")
        if key not in self.keys:
            raise KeyError(f"Key {key} not in list [{','.join(self.keys)}]")
