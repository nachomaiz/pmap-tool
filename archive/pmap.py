import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from prince.ca import CA
from prince import util, plot

class PMAP(CA):
    '''Perceptual Map object based on Correspondence Analysis.

    Implemented:
    __init__ : COMPLETE
    fit : COMPLETE
    fitted_supp_rows : COMPLETE
    fitted_supp_cols : COMPLETE
    plot_map : COMPLETE
    raw_data : COMPLETE
    fitted_data : COMPLETE

    '''
    def __init__(self, n_components: int = 2, n_iter: int = 10, copy: bool = True, check_input: bool = True, benzecri: bool = False,
                 random_state: int = None, engine: str = 'auto'):
        super().__init__(n_components=n_components, n_iter=n_iter, copy=copy, check_input=check_input, benzecri=benzecri,
                 random_state=random_state, engine=engine)
        
    def fit(self, X: pd.DataFrame, supp: tuple = None, y=None):
        '''Fits PMAP to dataframe, handling supplementary data separately.

        Parameters
        ----------
        X : dataframe to fit PMAP
            This dataframe should have cases by row and attributes by columns. Case labels should be reflected in the index.
        supp : None, tuple
            The number of supplementary rows and/or columns in the data. (supp_rows, supp_cols) Defaults to None.

        Returns
        -------
        new_PMAP : PMAP object fitted to X
        '''
        # check X type
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X is not a pandas DataFrame.')
        
        self.data = X.copy()
        self.supp = supp
        self.supp_rows = None
        self.supp_cols = None
        
        # check supp type
        if self.supp:
            if type(self.supp) == tuple:
                if any([type(i) != int or i < 0 for i in self.supp]):
                    raise ValueError('Supplementary rows and columns must be non-negative integers')
            else:
                raise TypeError('supp must be a tuple of non-negative integers')

            # cut and store supplementary data
            self.core = X.iloc[0:(X.shape[0] - self.supp[0]),
                                  0:(X.shape[1] - self.supp[1])]
            if r := self.supp[0]:
                self.supp_rows = X.iloc[-r:]
            if c := self.supp[1]:
                self.supp_cols = X.iloc[:,-c:]
        else:
            self.core = self.data
        
        # pass core data to CA fit method
        super().fit(X=self.core, y=y)

        return self

    def _make_MultiIndex(self) -> pd.MultiIndex:
        '''Returns rows and columns multi-index for formatted tables'''
        row_idx = tuple(zip(['Core' for _ in self.core.index] + ['Supplementary' for _ in range(self.supp[0])], self.data.index))
        row_idx = pd.MultiIndex.from_tuples(row_idx)
        
        col_idx = tuple(zip(['Core' for _ in self.core.columns] + ['Supplementary' for _ in range(self.supp[1])], self.data.columns))
        col_idx = pd.MultiIndex.from_tuples(col_idx)

        return row_idx, col_idx

    @property
    def fitted_supp_rows(self) -> pd.DataFrame:
        if self.supp_rows is not None:
            r = self.supp_rows if self.supp_cols is None else self.supp_rows.iloc[:,:-self.supp[1]]
            return self.row_coordinates(r)
        else: return None

    @property
    def fitted_supp_cols(self) -> pd.DataFrame:
        if self.supp_cols is not None:
            c = self.supp_cols if self.supp_rows is None else self.supp_cols.iloc[:-self.supp[0],:]
            return self.column_coordinates(c)
        else: return None

    @property
    def _fitted_tuple(self) -> tuple:
        '''Tuple of fitted data chunks for plotting loop'''
        row_label, _, col_label, _ = util.make_labels_and_names(self.data)
        sr = self.fitted_supp_rows.rename_axis("Supp " + row_label) if self.fitted_supp_rows is not None else None
        sc = self.fitted_supp_cols.rename_axis("Supp " + col_label) if self.fitted_supp_cols is not None else None
        return (self.row_coordinates(self.core).rename_axis(row_label), self.column_coordinates(self.core).rename_axis(col_label), 
                sr, sc)
    
    @property
    def raw_data(self) -> pd.DataFrame:
        '''The model's fitted data formatted with multi-index if supp data exists'''
        if self.supp:
            row_idx, col_idx = self._make_MultiIndex()
            return pd.DataFrame(self.data.values, row_idx, col_idx)
        else:
            return self.core

    @property
    def fitted_data(self) -> pd.DataFrame:
        '''The model's fitted data formatted with multi-index'''
        if self.supp:
            row_idx, col_idx = self._make_MultiIndex()
            r = self.row_coordinates(self.core)
            if self.supp[0] > 0:
                r = r.append(self.fitted_supp_rows, ignore_index=True)
            c = self.column_coordinates(self.core)
            if self.supp[1] > 0:
                c = c.append(self.fitted_supp_cols, ignore_index=True)
                        
            return pd.concat([r.set_index(row_idx), c.set_index(col_idx)], keys=['Rows','Columns'])
        else:
            return pd.concat([self.row_coordinates(self.core), self.column_coordinates(self.core)], 
                             keys=['Rows','Columns'])

    def _plot(self, X, ax, labels = True, only_labels = False, **kwargs):
        label, names, _, _ = util.make_labels_and_names(X)
        
        # Plot coordinates
        x = X.iloc[:,0]
        y = X.iloc[:,1]
        scatter = ax.scatter(x, y, label=label, **kwargs)

        if only_labels:
            c = ax.get_legend_handles_labels()[0][0].get_facecolor()[0]
            scatter.remove()

        # Add labels
        if labels:
            for xi, yi, lab in zip(x, y, names):
                if only_labels:
                    ax.annotate(lab, (xi, yi), ha='center', va='center', color=c)
                else:
                    ax.annotate(lab, (xi, yi))

    def plot_map(self, figsize: tuple = (16,9), ax = None, x_component: int = 0, y_component: int = 1,
                         supp = False, show_labels: tuple = (True,True), only_labels: bool = False, invert_ax: str = None, stylize=True, **kwargs):
        '''Plots perceptual map from trained self. 

        Parameters
        ----------
        figsize : tuple(int)
            Size of the returned plot. Ignored if ax is not None
        ax : matplotlib Axis, default = None
            The axis to plot into. Defaults to None, creating a new ax
        x_component, y_component : int
            Component from the trained self to use as x and y axis in the perceptual map
        supp : bool, 'only'
            Plot supplementary data (if present). 'only' will suppress core data and show supplementary data instead.
        show_labels : tuple(bool)
            (bool, bool) = show labels for [rows, columns]. If supp = True, shows all rows or columns
            (bool, bool, bool, bool) = only if supp == True, show labels for [rows, columns, supp rows, supp columns]
        invert_ax : str, default = None
            'x' = invert x axis 
            'y' = invert y axis 
            'b' = invert both axis
        **kwargs 
            Additional arguments to pass to matplotlib function
        
        Returns
        -------
        ax : matplotlib axis
            Perceptual map plot
        '''

        self._check_is_fitted()

        # Build figure if none is passed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Add style
        if stylize:
            ax = plot.stylize_axis(ax, grid=False)

        tup = self._fitted_tuple

        # Change inputs based on function parameters
        # if supp == "only" or supp == False all tuples are len 2, if supp == True all tuples are len 4
        if supp in ('only', False):
            if len(show_labels) != 2:
                raise ValueError('Length of show_labels expected to be 2')
            tup = tup[2:4] if supp == 'only' else tup[0:2]
        elif supp == True:
            if len(show_labels) not in (2,4):
                raise ValueError('Length of show_labels expected to be 2 or 4')
            show_labels = show_labels * 2 if len(show_labels) == 2 else show_labels
        else:
            raise ValueError("supp must be True, False or 'only'")

        # for only_labels
        legend_handles = []
        
        # Main plotting loop
        for i, l in zip(tup, show_labels):
            if i is not None:
                self._plot(i.loc[:,[x_component, y_component]], ax, l, only_labels, **kwargs)
                if only_labels and l:
                    c = next(ax._get_lines.prop_cycler)['color']
                    legend_handles.append(mpatches.Patch(color=c, label=i.index.name))

        # Manually build legend if only_labels is True
        if only_labels:
            ax.legend(handles=legend_handles)
        else:
            ax.legend()

        # Text
        ei = self.explained_inertia_
        ax.set_title('Principal Coordinates ({:.2f}% total inertia)'.format(100 * (ei[y_component]+ei[x_component])))
        ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]))
        ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]))

        # Invert axis if passed
        if invert_ax is not None:
            if invert_ax == 'x':
                ax.invert_xaxis()
            elif invert_ax == 'y':
                ax.invert_yaxis()
            elif invert_ax == 'b':
                ax.invert_xaxis()
                ax.invert_yaxis()
            else:
                raise ValueError("invert_ax must be 'x', 'y' or 'b' for both")
        
        return ax
    
    def plot_subplots(self, figsize: tuple = (16,9), axs = None, x_component: int = 0,
                y_component: int = 1, **kwargs):
        '''Plots each chunk of data into a separate subplot.
        
        Parameters
        ----------
        figsize : tuple(int)
            Size of the returned plot. Ignored if axes is not None
        axs : matplotlib Axes, default = None
            The axes to plot into. Defaults to None, creating a new axes
        x_component, y_component : int
            Component from the trained self to use as x and y axis in the perceptual map
        invert_ax : str, default = None
            'x' = invert x axis 
            'y' = invert y axis 
            'b' = invert both axis
        **kwargs 
            Additional arguments to pass to matplotlib function
        
        Returns
        -------
        axes : matplotlib axes
            Perceptual map plot
        '''

        self._check_is_fitted()

        chunks = tuple(c.loc[:,[x_component, y_component]] if c is not None else None for c in self._fitted_tuple)
        n_plots = len([c for c in chunks if c is not None])

        # Build figure if none is passed
        if axs is None:
            fig, axs = plt.subplots(nrows=2 if n_plots > 2 else 1, ncols=2, sharex=True, sharey=True, figsize=figsize, constrained_layout=True)
        
        colors = plt.cm.Dark2(range(0,4 if n_plots > 2 else 2))

        # Main plotting loop
        for ax, s, c in zip(axs.flat, chunks, colors):
            if s is not None:
                ax = plot.stylize_axis(ax, grid=False) # Add style
                self._plot(s, ax, color=c, **kwargs)
                ax.legend()
        
        # Hide empty plots
        if n_plots == 3:
            for ax in axs.flat:
                if not ax.lines:
                    ax.remove()
        
        # Text
        ei = self.explained_inertia_
        fig.suptitle('Principal Coordinates ({:.2f}% total inertia)'.format(100 * (ei[1]+ei[0])), fontsize=16)
        fig.supxlabel('Component {} ({:.2f}% inertia)'.format(0, 100 * ei[0]))
        fig.supylabel('Component {} ({:.2f}% inertia)'.format(1, 100 * ei[1]))

        return axs