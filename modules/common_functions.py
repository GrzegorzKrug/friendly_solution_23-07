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


if __name__ == "__main__":
    nm = NamingClass("pierwszy", "lstm", "1", "10","500", "1000", "3", "0.5")
    print(nm)
    print(nm.path)

    name2 = NamingClass.read_from_path(nm.path)
    print(name2)
    print(name2.path)

    print(nm.path == name2.path)
