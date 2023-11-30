import os
import pandas as pd


class Report:
    def __init__(self, nodes, export=True, filename="report"):
        self.nodes = nodes
        self.export = export
        self.filename = filename

    def get_stats(self, topk='top1', verbose=False):

        df = pd.DataFrame(
                index=[node.name for node in self.nodes], columns=[
                'weight', 'bias', 'input_shape', 'output_shape', 'in_tensors', 'out_tensors',
                'active_blocks', 'ram']
                )

        for node in self.nodes:
            df.weight[node.name] = node.weights
            df.bias[node.name] = node.bias
            df.input_shape[node.name] = [x.shape for x in node.inputs]
            df.output_shape[node.name] = [x.shape for x in node.outputs]
            df.in_tensors[node.name] = [x.name for x in node.inputs]
            df.out_tensors[node.name] = [x.name for x in node.outputs]
            df.active_blocks[node.name] = node.malloc_blocks
            df.ram[node.name] = node.malloc_val

        if verbose:
            idx_max = df.index[df.ram == df.ram.max()]
            print('-' * 120)
            print(df.to_string())
            print('-' * 120)
            print()
            print(" >> Peak Memory of {:.0f} kB found in the following node(s):".format(
                df.ram.max() / (2**10)))
            print()

            if topk == 'top1': # eventually to extend to top-n
                for idx in idx_max:
                    print(df.loc[idx].to_string())
                    print()
            print()

        if self.export:
            self.export_results(df)

        return df

    def export_results(self, df, plot=False):

        filename = self.filename # to fix
        if not os.path.isdir('results'):
            os.mkdir('results')
        outpath = os.path.join('results', filename)
        df.to_csv(outpath + ".csv", sep='\t', encoding='utf-8')
        print(" >> .csv summary results saved in {}".format(outpath + ".csv"))

        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize = (15, 5), dpi=300)
            ax = fig.add_axes([0,0,1,1])
            ind = [i for i in range(len(df))]
            ax.bar(ind, [ram / (2**20) for ram in df.ram.values], color='r')
            ax.set_ylabel('Memory [MB]')
            ax.set_title(filename)
            ax.grid()
            ax.set_xticks(ind)
            ax.set_xticklabels(df.index.values, rotation=75)
            plt.savefig(outpath + ".png", bbox_inches='tight', dpi=300)
            print(" >> .png barplot saved in {}".format(outpath + ".csv\n"))
