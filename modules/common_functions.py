from matplotlib import pyplot as plt
import numpy as np
import os


class NamingClass:
    PAR_SEP = ","
    VAL_SEP = '_'

    def __init__(
            self,
            model_name,
            arch_name,
            arch_series,
            feat_n,
            window_n,
            node_insize,
            node_outsize,
            reward_fnum,
            postfix="",

    ):
        self.model_name = str(model_name)
        self.arch_name = str(arch_name)
        self.arch_series = str(arch_series)

        self.feat_n = str(feat_n)
        self.window_n = str(window_n)

        self.node_insize = str(node_insize)
        self.node_outsize = str(node_outsize)

        self.reward_fnum = str(reward_fnum)
        self.postfix = postfix

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

    @property
    def path(self):
        text = f"model{self.VAL_SEP}{self.model_name}"
        text += f"{self.PAR_SEP}ar{self.VAL_SEP}{self.arch_name}"
        text += f"{self.PAR_SEP}ar{self.VAL_SEP}{self.arch_series}"
        text += f"{self.PAR_SEP}fn{self.VAL_SEP}{self.feat_n}"
        text += f"{self.PAR_SEP}wn{self.VAL_SEP}{self.window_n}"
        text += f"{self.PAR_SEP}ni{self.VAL_SEP}{self.node_insize}"
        text += f"{self.PAR_SEP}no{self.VAL_SEP}{self.node_outsize}"
        text += f"{self.PAR_SEP}rf{self.VAL_SEP}{self.reward_fnum}"
        text += f"{self.PAR_SEP}pf{self.VAL_SEP}{self.postfix}"

        return text

    def copy(self):
        return NamingClass(
                self.model_name,
                self.arch_series,
                self.arch_name,
                self.node_insize,
                self.node_outsize,
                self.reward_fnum,
                self.feat_n,
        )

    def __str__(self):
        return f"PathingFiles({self.path})"


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


if __name__ == "__main__":
    arr = np.array([
            [0, 0, 0, 0],
            [0.3, 0.3, 0.3, 0.3],
            [1, 10, 20, 30],
            [3, 30, 60, 90],
            [7, 70, 140, 210],
    ]
    )
    interp_2d(arr)