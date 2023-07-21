from matplotlib import pyplot as plt
import numpy as np
import os


class NamingClass:
    PAR_SEP = ","
    VAL_SEP = '_'

    def __init__(
            self,
            arch_series,
            iteration,

            time_feats,
            time_window,
            float_feats,
            outsize,
            node_size,

            learning_rate,
            loss,
            batch,

            optimizer="",
            reward_fnum="",
            discount="",
            postfix="",
    ):
        self.arch_series = str(arch_series)
        self.iteration = str(iteration)

        self.time_feats = str(time_feats)
        self.time_window = str(time_window)
        self.float_feats = str(float_feats)
        self.outsize = str(outsize)
        self.node_size = str(node_size)

        self.learning_rate = str(learning_rate)
        self.loss = str(loss)
        self.batch = str(batch)

        "Optional"
        self.optimizer = str(optimizer)
        self.reward_fnum = str(reward_fnum)
        self.discount = str(discount)
        self.postfix = str(postfix)

    @classmethod
    def read_from_path(cls, path, remove_ext=True):
        path = os.path.basename(path)

        if '.' in path[-5:] and remove_ext:
            path, _ = path.split('.')
            # print("Path no extension", path)

        args = path.split(cls.PAR_SEP)
        if args[0] == "model":
            args = args[1:]

        string_args = [a.split(cls.VAL_SEP)[1] for a in args if len(a) > 0]

        return cls(*string_args)

    @classmethod
    def from_path(cls, *a, **kw):
        return cls.read_from_path(*a, **kw)

    @property
    def path(self):
        text = f"model{self.VAL_SEP}{self.arch_series}"
        # text += f"{self.PAR_SEP}ar{self.VAL_SEP}{self.arch_series}"
        text += f"{self.PAR_SEP}it{self.VAL_SEP}{self.iteration}"

        text += f"{self.PAR_SEP}tf{self.VAL_SEP}{self.time_feats}"
        text += f"{self.PAR_SEP}tw{self.VAL_SEP}{self.time_window}"
        text += f"{self.PAR_SEP}ff{self.VAL_SEP}{self.float_feats}"

        text += f"{self.PAR_SEP}ou{self.VAL_SEP}{self.outsize}"
        text += f"{self.PAR_SEP}no{self.VAL_SEP}{self.node_size}"

        text += f"{self.PAR_SEP}lr{self.VAL_SEP}{self.learning_rate}"
        text += f"{self.PAR_SEP}ls{self.VAL_SEP}{self.loss}"
        text += f"{self.PAR_SEP}bt{self.VAL_SEP}{self.batch}"

        text += f"{self.PAR_SEP}op{self.VAL_SEP}{self.optimizer}"
        text += f"{self.PAR_SEP}rf{self.VAL_SEP}{self.reward_fnum}"
        text += f"{self.PAR_SEP}dc{self.VAL_SEP}{self.discount}"
        text += f"{self.PAR_SEP}pf{self.VAL_SEP}{self.postfix}"

        return text

    @property
    def cur_model_path(self):
        return self.path

    def copy(self):
        return NamingClass(
                self.arch_series,
                self.iteration,

                self.time_feats,
                self.time_window,
                self.float_feats,

                self.outsize,
                self.node_size,
                self.learning_rate,
                self.loss,
                self.batch,

                self.optimizer,
                self.reward_fnum,
                self.discount,
                self.postfix,
        )

    def __str__(self):
        return f"NamingClass({self.path})"


def interp_2d(arr):
    tm = arr[:, 0]
    vals = arr[:, 1:]
    # print("ARR:")
    # print(arr)
    # print()
    # print(tm)
    # print(vals)
    tm_uniform = np.arange(tm[0], tm[-1] + 0.1, 0.1)
    # tm_uniform = np.concatenate()

    # out_arr = np.empty(shape=(0,))
    out_arr = tm_uniform.reshape(-1, 1)
    # print("Empty shape:")
    # print(out_arr.shape)

    for col in vals.T:
        # print("Column", col)
        # print(col.shape, tm.shape)
        vals_uni = np.interp(tm_uniform, tm, col).reshape(-1, 1)
        out_arr = np.concatenate([out_arr, vals_uni], axis=1)

    # print(out_arr)
    # print(vals_uni)
    x1, y1 = (arr[:, [0, 1]].T)
    x2, y2 = (out_arr[:, [0, 1]].T)
    plt.plot(x1, y1)
    plt.plot(x2, y2, dashes=[2, 1])
    plt.show()


def interp_1d_sub(tm_uni, tm, vals):
    vals_uni = np.interp(tm_uni, tm, vals)
    return vals_uni


def to_sequences_forward(array, seq_size=1, fwd_intervals=[1]):
    x = []
    y = []
    offset_arr = np.array(fwd_intervals) - 1
    last_minus = max(fwd_intervals) - 1
    for i in range(len(array) - seq_size - last_minus):
        window = array[i:(i + seq_size), :]
        x.append(window)
        sub_arr = array[i + seq_size + offset_arr, :]
        y.append(sub_arr)
    return np.array(x, dtype=np.float16), np.array(y, np.float16)


def to_sequences_forward_keep_features(array, seq_size=1, fwd_intervals=[1], ft_amount=0):
    x = []
    y = []
    offset_arr = np.array(fwd_intervals) - 1
    last_minus = max(fwd_intervals) - 1
    columns = array.shape[1]
    lstm_cols = columns - ft_amount
    for i in range(len(array) - seq_size - last_minus):
        window = array[i:(i + seq_size), :]
        x.append(window)
        sub_arr = array[i + seq_size + offset_arr, :lstm_cols]
        y.append(sub_arr)
    return np.array(x), np.array(y)


def get_splits(data_size=5000, split_size=100):
    segments = data_size // split_size

    inds = [0]
    for i in range(1, segments):
        inds.append(i * split_size)

    if inds[-1] != data_size + 1:
        inds.append(data_size + 1)

    return inds


def get_eps(n, epoch_max, repeat=9, eps_power=1.4, max_explore=0.8):
    """
    Works only with matching periods, last period may be be different if not
    Condition: epoch_max % repeat == 0

    Args:
        n:
        epoch_max:
        repeat:
        eps_power:
        max_explore:

    Returns:

    """
    "N is -1 smaller than epoch_max"
    if n >= epoch_max:
        raise ValueError("N can not be equal to epoch max!")
    disc_step = (epoch_max) // repeat

    if epoch_max <= 25:
        val = ((1 - n / (epoch_max - 1)) ** eps_power) * max_explore
    else:
        return (1 - np.mod(n, disc_step) / (disc_step - 1)) ** eps_power * max_explore
    return val

def load_data_split(path, train_split=0.65, ):
    arr = np.load(path, allow_pickle=True)

    pivot = int(len(arr) * train_split)

    df_train = arr[:pivot, :]
    df_test = arr[pivot:, :]
    print(f"Train: {df_train.shape}, Test: {df_test.shape}")
    return df_train, df_test

if __name__ == "__main__":
    M = 50
    X = np.arange(M)

    for i in [1, 49, 50, 51, 99]:
        print(i, f"{get_eps(i, 100, repeat=2):>4.4f}")

    Y = [get_eps(x, M, repeat=5) for x in X]
    Y = [get_eps(x, M, repeat=25, eps_power=1, max_explore=1) for x in X]
    plt.plot(X, Y)
    plt.grid()
    plt.tight_layout()
    plt.show()
