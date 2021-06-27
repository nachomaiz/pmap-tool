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
            else:
                self.supp_rows = None
            if self.supp[1]:
                self.supp_cols = X.iloc[:,-self.supp[1]:]
            else:
                self.supp_cols = None
        else:
            self.core = self.data
        
        # pass core data to CA fit method
        super().fit(X=self.core, y=y)

        return self

    def _make_MultiIndex(self) -> pd.MultiIndex:
        '''Returns rows and columns multi-index for formatted tables'''
        _row_idx = tuple(zip(['Core' for c in self.core.index] + ['Supplementary' for c in range(self.supp[0])], self.data.index))
        _row_idx = pd.MultiIndex.from_tuples(_row_idx)
        
        _col_idx = tuple(zip(['Core' for c in self.core.columns] + ['Supplementary' for c in range(self.supp[1])], self.data.columns))
        _col_idx = pd.MultiIndex.from_tuples(_col_idx)

        return _row_idx, _col_idx

    @property
    def fitted_supp_rows(self) -> pd.DataFrame:
        if self.supp is None:
            return None
        if self.supp_rows is not None:
            if self.supp[1] > 0:
                return self.row_coordinates(self.supp_rows.iloc[:,:-self.supp[1]])
            else:
                return self.row_coordinates(self.supp_rows)
        else: return None

    @property
    def fitted_supp_cols(self) -> pd.DataFrame:
        if self.supp is None:
            return None
        if self.supp_cols is not None:
            if self.supp[0] > 0:
                return self.column_coordinates(self.supp_cols.iloc[:-self.supp[0],:])
            else:
                return self.column_coordinates(self.supp_cols)
        else: return None

    @property
    def _fitted_tuple(self) -> tuple:
        '''Tuple of fitted data chunks for plotting loop'''
        t = [self.row_coordinates(self.core), self.column_coordinates(self.core)]
        if self.supp is not None:
            if self.supp[0] > 0:
                sr = self.fitted_supp_rows
            else: sr = None

            if self.supp[1] > 0:
                sc = self.fitted_supp_cols
            else: sc = None
            
            t.extend([sr, sc])
        return tuple(t)
    
    @property
    def raw_data(self) -> pd.DataFrame:
        '''The model's fitted data formatted with multi-index if supp data exists'''
        if self.supp:
            _row_idx, _col_idx = self._make_MultiIndex()
            return pd.DataFrame(self.data.values, _row_idx, _col_idx)
        else:
            return self.core

    @property
    def fitted_data(self) -> pd.DataFrame:
        '''The model's fitted data formatted with multi-index'''
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
    
    def get_chart_data(self, x_component:int = 0, y_component:int = 1, invert_ax: str = None) -> pd.DataFrame:
        '''Returns two dimensions of multi-indexed, invert-compatible fitted data'''
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

    def _plot(self, X, ax, axis: int = 0, supp: bool = False, labels: bool = True, **kwargs):
        _, names, _, _ = util.make_labels_and_names(X)
        
        # Parse label accounting for supplementary data
        if axis == 0:
            label = "Rows"
        elif axis == 1:
            label = "Columns"
        else:
            raise ValueError("axis must be 0 or 1")
        
        label = 'Supp ' + label if supp else label

        # Plot coordinates
        x = X.iloc[:,0]
        y = X.iloc[:,1]
        ax.scatter(x, y, **kwargs, label=label)

        # Add labels
        if labels:
            for xi, yi, lab in zip(x, y, names):
                ax.annotate(lab, (xi, yi))

        return ax

    def plot_map(self, figsize: tuple = (16,9), ax = None, x_component: int = 0, y_component: int = 1,
                         supp = False, show_labels: tuple = (True,True), invert_ax: str = None, **kwargs):
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
        ax = plot.stylize_axis(ax, grid=False)

        tup = self._fitted_tuple
        pl_supp = (False, False)

        # Change inputs based on function parameters
        # if supp == "only" or supp == False all tuples are len 2, if supp == True all tuples are len 4
        if supp == 'only' or supp == False:
            if len(show_labels) != 2:
                raise ValueError('Length of show_labels expected to be 2')
            if supp == 'only':
                tup = tup[2:4] # only uses supp data
                pl_supp = (True, True)
            if supp == False:
                tup = tup[:2] # only uses core data
        elif supp == True:
            if len(show_labels) != 4 and len(show_labels) != 2:
                raise ValueError('Length of show_labels expected to be 2 or 4')
            else :
                show_labels = show_labels * 2 if len(show_labels) == 2 else show_labels
            pl_supp = pl_supp + (True, True)
        else:
            raise ValueError("supp must be True, False or 'only'")            

        # Main plotting loop
        count = 0
        for i in tup:
            if i is not None:
                axis = count % 2 # 0 if rows 1 if cols
                ax = self._plot(i.loc[:,[x_component, y_component]], ax, axis, pl_supp[count], show_labels[count], **kwargs)
            count += 1

        # Legend
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
    
    def subplots(self, figsize: tuple = (16,9), axes = None, x_component: int = 0,
                y_component: int = 1, **kwargs):
        '''Plots each chunk of data into a separate subplot.
        
        Parameters
        ----------
        figsize : tuple(int)
            Size of the returned plot. Ignored if axes is not None
        axes : matplotlib Axes, default = None
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

        chunks = [c.loc[:,[x_component, y_component]] for c in self._fitted_tuple if c is not None]
        n_plots = len(chunks)

        # Build figure if none is passed
        if axes is None:
            fig, axes = plt.subplots(nrows=2, ncols=2 if n_plots > 2 else 1, sharex=True, sharey=True, figsize=figsize, constrained_layout=True)
        
        colors = plt.cm.Dark2(range(0,len(chunks)))

        # Main plotting loop
        count = 0
        for ax, s, c in zip(axes.flat, self._fitted_tuple, colors):
            if s is not None:
                axis = count % 2 # 0 if rows 1 if cols
                ax = plot.stylize_axis(ax, grid=False) # Add style
                self._plot(s, ax, axis, True if count > 1 else False, color=c, **kwargs)
                ax.legend()
                count += 1
        
        # Text
        ei = self.explained_inertia_
        fig.suptitle('Principal Coordinates ({:.2f}% total inertia)'.format(100 * (ei[1]+ei[0])), fontsize=16)
        fig.supxlabel('Component {} ({:.2f}% inertia)'.format(0, 100 * ei[0]))
        fig.supylabel('Component {} ({:.2f}% inertia)'.format(1, 100 * ei[1]))

        return axes