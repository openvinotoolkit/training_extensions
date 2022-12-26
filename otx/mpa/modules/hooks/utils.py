# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from matplotlib import pyplot as plt


def print_report(df, exp, logger=None):
    df_exp = df[df.exp == exp]
    df_pprint = (
        df_exp.assign(open_layer=lambda ddf: ddf.hook_type.map(lambda x: {"pre": 0, "fwd": 1, "bwd": 2}[x])
                      .rolling(2)
                      .apply(lambda x: list(x)[0] == 0 and list(x)[1] == 0))
        .assign(close_layer=lambda ddf: ddf.hook_type.map(lambda x: {"pre": 0, "fwd": 1, "bwd": 2}[x])
                .rolling(2).apply(lambda x: list(x)[0] == 1 and list(x)[1] == 1))
        .assign(indent_level=lambda ddf: (ddf.open_layer.cumsum() - ddf.close_layer.cumsum()).fillna(0).map(int))
        .sort_values(by="call_idx")
        .assign(mem_diff=lambda ddf: ddf.mem_all.diff() // 2 ** 20)
    )
    pprint_lines = [
        f"{row[1].call_idx:05}:{'    ' * row[1].indent_level}{row[1].layer_type} {row[1].hook_type}  \
            {row[1].mem_diff or ''}"
        for row in df_pprint.iterrows()
    ]
    logger.info('*' * 55)
    logger.info(f"{'*' * 5}  Memory allocation results for each layers  {'*' * 5}")
    logger.info('*' * 55)
    for x in pprint_lines:
        if logger is not None:
            logger.info(x)
        else:
            print(x)
    logger.info('*' * 55)
    logger.info('*' * 55)


def plot_mem(
        df,
        exps=None,
        normalize_call_idx=True,
        normalize_mem_all=True,
        filter_fwd=False,
        return_df=False,
        output_file=None):
    if exps is None:
        exps = df.exp.drop_duplicates()

    fig, ax = plt.subplots(figsize=(20, 10))
    for exp in exps:
        df_ = df[df.exp == exp]

        if normalize_call_idx:
            df_.call_idx = df_.call_idx / df_.call_idx.max()

        if normalize_mem_all:
            df_.mem_all = df_.mem_all - df_[df_.call_idx == df_.call_idx.min()].mem_all.iloc[0]
            df_.mem_all = df_.mem_all // 2 ** 20

        if filter_fwd:
            layer_idx = 0
            callidx_stop = df_[(df_["layer_idx"] == layer_idx) & (df_["hook_type"] == "fwd")]["call_idx"].iloc[0]
            df_ = df_[df_["call_idx"] <= callidx_stop]
            # df_ = df_[df_.call_idx < df_[df_.layer_idx=='bwd'].call_idx.min()]

        plot = df_.plot(ax=ax, x='call_idx', y='mem_all', label=exp)
        if output_file:
            plot.get_figure().savefig(output_file)

    if return_df:
        return df_
