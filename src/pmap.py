import matplotlib as mpl
import matplotlib.pyplot as plt
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
    plot_coordinates : COMPLETE
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
        -----------
        X : dataframe to fit self with
            This dataframe should have cases by row and attributes by columns. Case labels should be reflected in the index.
        supp : None, tuple
            The number of supplementary rows and/or columns in the data. (supp_rows, supp_cols) Defaults to None.

        '''
        # check X type
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X is not a pandas DataFrame.')
        
        self.data = X.copy()
        self.supp = supp
        
        # cut out and store supplementary data
        if self.supp:
            # check supp type
            if type(self.supp) == tuple:
                if any([type(i) != int or i < 0 for i in self.supp]):
                    raise ValueError('Supplementary rows and columns must be non-negative integers')
            else:
                raise TypeError('supp must be a tuple of non-negative integers')

            # cut and store supplementary data
            self.core = X.iloc[0:(X.shape[0] - self.supp[0]),
                                  0:(X.shape[1] - self.supp[1])]
            if self.supp[0]:
                self.supp_rows = X.iloc[-self.supp[0]:]
            if self.supp[1]:
                self.supp_cols = X.iloc[:,-self.supp[1]:]
        else:
            self.core = self.data
        
        # pass core data to CA fit method
        super().fit(X=self.core, y=y)

        return self

    @property
    def fitted_supp_rows(self) -> pd.DataFrame:
        if self.supp[1] > 0:
            return self.row_coordinates(self.supp_rows.iloc[:,:-self.supp[1]])
        else:
            return self.row_coordinates(self.supp_rows)

    @property
    def fitted_supp_cols(self) -> pd.DataFrame:
        if self.supp[0] > 0:
            return self.column_coordinates(self.supp_cols.iloc[:-self.supp[0],:])
        else:
            return self.column_coordinates(self.supp_cols)

    def plot_supp(self, figsize: tuple = (16,9), ax = None, x_component: int = 0, y_component: int = 1, show_labels: tuple = (True,True), **kwargs):
        '''Refactoring of plot_coordinates to plot supplementary data'''

        self._check_is_fitted()

        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)
        
        # Add style
        ax = plot.stylize_axis(ax)

        # Get labels and names
        if self.supp[0] > 0:
            row_label, row_names, _, _ = util.make_labels_and_names(self.supp_rows)
            # Plot row principal coordinates (minus supp cols)
            row_coords = self.fitted_supp_rows
            ax.scatter(
                row_coords[x_component],
                row_coords[y_component],
                **kwargs,
                label='Supp ' + row_label
            )

            # Add row labels
            if show_labels[0]:
                x = row_coords[x_component]
                y = row_coords[y_component]
                for xi, yi, label in zip(x, y, row_names):
                    ax.annotate(label, (xi, yi))
        
        if self.supp[1] > 0:
            _, _, col_label, col_names = util.make_labels_and_names(self.supp_cols)

            # Plot column principal coordinates (minus supp rows)
            col_coords = self.fitted_supp_cols
            ax.scatter(
                col_coords[x_component],
                col_coords[y_component],
                **kwargs,
                label='Supp ' + col_label
            )

            # Add column labels
            if show_labels[1]:
                x = col_coords[x_component]
                y = col_coords[y_component]
                for xi, yi, label in zip(x, y, col_names):
                    ax.annotate(label, (xi, yi))

        # Legend
        ax.legend()
        ax.grid(False) # force grid off

        # Text
        ei = self.explained_inertia_
        ax.set_title('Principal Coordinates ({:.2f}% total inertia)'.format(100 * (ei[y_component]+ei[x_component])))
        ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]))
        ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]))

        return ax

    def plot_coordinates(self, figsize = (16,9), ax = None, x_component = 0, y_component = 1,
                         supp = False, show_labels = (True,True), invert_ax = None, **kwargs):
        
        '''Plots perceptual map from trained self. 

        Parameters
        ----------
        figsize : tuple(int)
            Size of the returned plot
        ax : matplotlib Axis, default = None
            The axis to plot into. Defaults to None, creating a new ax
        x_component, y_component : int
            Component from the trained self to use as x and y axis in the perceptual map
        supp : bool, 'only'
            Plot supplementary data (if present). 'only' will suppress core data and show supplementary data instead.
        show_labels : tuple(bool)
            (bool, bool) : show labels for [rows, columns]. If supp = True, shows all rows or columns
            (bool, bool, bool, bool) : only if supp = True, show labels for [rows, columns, supp rows, supp columns]
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

        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)
        
        # Add style
        ax = plot.stylize_axis(ax)

        # check supp and show_labels inputs
        if supp == 'only' or supp == False:
            if len(show_labels) != 2:
                raise ValueError('Length of show_labels expected to be 2')
        elif supp == True:
            if len(show_labels) != 4 and len(show_labels) != 2:
                raise ValueError('Length of show_labels expected to be 2 or 4')
        else:
            raise ValueError("supp must be True, False or 'only'")

        # if 'only', will need to plot into ax not from plot_coordinates, otherwise plot normally and add supp if needed
        if supp == 'only':
            
            ax = self.plot_supp(figsize=figsize, ax=ax, x_component=x_component, y_component=y_component, show_labels=show_labels, **kwargs)

        else:
            ax = super().plot_coordinates(X = self.core, 
                                        figsize = figsize,
                                        x_component = x_component,
                                        y_component = y_component,
                                        show_row_labels = show_labels[0], 
                                        show_col_labels = show_labels[1], 
                                        ax = ax,
                                        **kwargs)
            
            if supp == True:
                if len(show_labels) == 2:
                    show_labels = show_labels * 2
                self.plot_supp(ax=ax, x_component=x_component, y_component=y_component, show_labels = (show_labels[2],show_labels[3]))
            
            ei = self.explained_inertia_
            ax.set_title('Principal Coordinates ({:.2f}% total inertia)'.format(100 * (ei[y_component]+ei[x_component])))

        if invert_ax is not None:
            if invert_ax == 'x':
                ax.invert_xaxis()
            elif invert_ax == 'y':
                ax.invert_yaxis()
            elif invert_ax == 'b':
                ax.invert_xaxis()
                ax.invert_yaxis()
            else:
                raise ValueError("invert must be 'x', 'y' or 'b' for both")

        return ax

    def _make_MultiIndex(self) -> pd.MultiIndex:
        _row_idx = tuple(zip(['Core' for c in self.core.index] + ['Supplementary' for c in range(self.supp[0])], self.data.index))
        _row_idx = pd.MultiIndex.from_tuples(_row_idx)
        
        _col_idx = tuple(zip(['Core' for c in self.core.columns] + ['Supplementary' for c in range(self.supp[1])], self.data.columns))
        _col_idx = pd.MultiIndex.from_tuples(_col_idx)

        return _row_idx, _col_idx

    @property
    def raw_data(self) -> pd.DataFrame:
        if self.supp:
            _row_idx, _col_idx = self._make_MultiIndex()
            return pd.DataFrame(self.data.values, _row_idx, _col_idx)
        else:
            return self.core

    @property
    def fitted_data(self) -> pd.DataFrame:
        if self.supp:
            _row_idx, _col_idx = self._make_MultiIndex()
            _r = self.row_coordinates(self.core)
            if self.supp[0] > 0:
                _r = _r.append(self.fitted_supp_rows, ignore_index=True)
            _c = self.column_coordinates(self.core)
            if self.supp[1] > 0:
                _c = _c.append(self.fitted_supp_cols, ignore_index=True)
                        
            return pd.concat([_r.set_index(_row_idx), _c.set_index(_col_idx)], keys=['Rows','Columns'])
        else:
            return pd.concat([self.row_coordinates(self.core), 
                              self.column_coordinates(self.core)], 
                              keys=['Rows','Columns'])
    
    def get_chart_data(self, x_component:int = 0, y_component:int = 1, invert_ax=None) -> pd.DataFrame:
        _d = self.fitted_data[[x_component, y_component]]
        if invert_ax is not None:
            if invert_ax == 'x':
                _d[x_component] = _d[x_component] * -1
            elif invert_ax == 'y':
                _d[y_component] = _d[y_component] * -1
            elif invert_ax == 'b':
                _d = _d * -1
            else:
                raise ValueError("invert_ax must be 'x', 'y' or 'b' for both")
        
        return _d