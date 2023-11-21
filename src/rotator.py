import numpy as np
import pandas as pd
from factor_analyzer.rotator import (
    OBLIQUE_ROTATIONS,
    ORTHOGONAL_ROTATIONS,
    POSSIBLE_ROTATIONS,
    Rotator,
)

ORTH_ROTATIONS = ["varimax", "oblimax", "quartimax", "equamax", "geomin_ort"]

OBL_ROTATIONS = ["oblimin", "quartimin", "geomin_obl"]

ROTATIONS = ORTH_ROTATIONS + OBL_ROTATIONS


class PmapRotator(Rotator):
    """
    The Rotator class takes an (unrotated)
    factor loading matrix and performs one
    of several rotations.

    Parameters
    ----------
    method : str, optional
        The factor rotation method. Options include:
            (a) varimax (orthogonal rotation)
            (b) promax (oblique rotation)
            (c) oblimin (oblique rotation)
            (d) oblimax (orthogonal rotation)
            (e) quartimin (oblique rotation)
            (f) quartimax (orthogonal rotation)
            (g) equamax (orthogonal rotation)
            (h) geomin_obl (oblique rotation)
            (i) geomin_ort (orthogonal rotation)
        Defaults to 'varimax'.
    normalize : bool or None, optional
        Whether to perform Kaiser normalization
        and de-normalization prior to and following
        rotation. Used for varimax and promax rotations.
        If None, default for promax is False, and default
        for varimax is True.
        Defaults to None.
    power : int, optional
        The power to which to raise the promax loadings
        (minus 1). Numbers should generally range form 2 to 4.
        Defaults to 4.
    kappa : int, optional
        The kappa value for the equamax objective.
        Ignored if the method is not 'equamax'.
        Defaults to 0.
    gamma : int, optional
        The gamma level for the oblimin objective.
        Ignored if the method is not 'oblimin'.
        Defaults to 0.
    delta : float, optional
        The delta level for geomin objectives.
         Ignored if the method is not 'geomin_*'.
        Defaults to 0.01.
    max_iter : int, optional
        The maximum number of iterations.
        Used for varimax and oblique rotations.
        Defaults to `1000`.
    tol : float, optional
        The convergence threshold.
        Used for varimax and oblique rotations.
        Defaults to `1e-5`.

    Attributes
    ----------
    loadings_ : numpy array, shape (n_features, n_factors)
        The loadings matrix
    rotation_ : numpy array, shape (n_factors, n_factors)
        The rotation matrix
    psi_ : numpy array or None
        The factor correlations
        matrix. This only exists
        if the rotation is oblique.

    Notes
    -----
    Most of the rotations in this class
    are ported from R's `GPARotation` package.

    References
    ----------
    [1] https://cran.r-project.org/web/packages/GPArotation/index.html

    Examples
    --------
    >>> import pandas as pd
    >>> from factor_analyzer import FactorAnalyzer, Rotator
    >>> df_features = pd.read_csv('test02.csv')
    >>> fa = FactorAnalyzer(rotation=None)
    >>> fa.fit(df_features)
    >>> rotator = Rotator()
    >>> rotator.fit_transform(fa.loadings_)
    array([[-0.07693215,  0.04499572,  0.76211208],
           [ 0.01842035,  0.05757874,  0.01297908],
           [ 0.06067925,  0.70692662, -0.03311798],
           [ 0.11314343,  0.84525117, -0.03407129],
           [ 0.15307233,  0.5553474 , -0.00121802],
           [ 0.77450832,  0.1474666 ,  0.20118338],
           [ 0.7063001 ,  0.17229555, -0.30093981],
           [ 0.83990851,  0.15058874, -0.06182469],
           [ 0.76620579,  0.1045194 , -0.22649615],
           [ 0.81372945,  0.20915845,  0.07479506]])
    """

    def _oblique_transform(self, loadings: pd.DataFrame) -> pd.DataFrame:
        """
        A generic function for performing
        all oblique rotations, except for
        promax, which is implemented
        separately.

        Parameters
        ----------
        loadings : array-like
            The loading matrix

        Returns
        -------
        loadings : numpy array, shape (n_features, n_factors)
            The loadings matrix
        """

        return np.dot(loadings, np.linalg.inv(self.rotation_).T)

    def _orthogonal_transform(self, loadings: pd.DataFrame) -> pd.DataFrame:
        """
        A generic function for performing
        all orthogonal rotations, except for
        varimax, which is implemented
        separately.

        Parameters
        ----------
        loadings : numpy array
            The loading matrix

        Returns
        -------
        loadings : numpy array, shape (n_features, n_factors)
            The loadings matrix
        """

        return np.dot(loadings, self.rotation_)

    def _varimax_transform(self, loadings: pd.DataFrame) -> pd.DataFrame:
        """
        Perform varimax (orthogonal) rotation, with optional
        Kaiser normalization.

        Parameters
        ----------
        loadings : array-like
            The loading matrix

        Returns
        -------
        loadings : numpy array, shape (n_features, n_factors)
            The loadings matrix
        rotation_mtx : numpy array, shape (n_factors, n_factors)
            The rotation matrix
        """
        n_cols = loadings.shape[1]
        if n_cols < 2:
            return loadings

        # normalize the loadings matrix
        # using sqrt of the sum of squares (Kaiser)
        if self.normalize:
            rot_mtx = np.apply_along_axis(
                lambda x: np.sqrt(np.sum(x**2)), 1, loadings
            )
            new_loadings = (loadings.T / rot_mtx).T
            new_loadings = np.dot(new_loadings, self.rotation_)
            new_loadings = (new_loadings.T * rot_mtx).T
            return new_loadings

        return np.dot(loadings, self.rotation_)

    def _promax_transform(self, loadings: pd.DataFrame) -> pd.DataFrame:
        """
        Perform promax (oblique) rotation, with optional
        Kaiser normalization.

        Parameters
        ----------
        loadings : array-like
            The loading matrix

        Returns
        -------
        loadings : numpy array, shape (n_features, n_factors)
            The loadings matrix
        rotation_mtx : numpy array, shape (n_factors, n_factors)
            The rotation matrix
        psi : numpy array or None, shape (n_factors, n_factors)
            The factor correlations
            matrix. This only exists
            if the rotation is oblique.
        """
        X = loadings.copy()
        n_cols = X.shape[1]
        if n_cols < 2:
            return X

        if self.normalize:
            # pre-normalization is done in R's
            # `kaiser()` function when rotate='Promax'.
            array = X.copy()
            h2 = np.diag(np.dot(array, array.T))
            h2 = np.reshape(h2, (h2.shape[0], 1))
            weights = array / np.sqrt(h2)

        else:
            weights = X.copy()

        # first get varimax rotation
        X, rotation_mtx = self._varimax(weights)
        Y = X * np.abs(X) ** (self.power - 1)

        # fit linear regression model
        coef = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

        # calculate diagonal of inverse square
        try:
            diag_inv = np.diag(np.linalg.inv(np.dot(coef.T, coef)))
        except np.linalg.LinAlgError:
            diag_inv = np.diag(np.linalg.pinv(np.dot(coef.T, coef)))

        # transform and calculate inner products
        coef = np.dot(coef, np.diag(np.sqrt(diag_inv)))
        z = np.dot(X, coef)

        if self.normalize:
            # post-normalization is done in R's
            # `kaiser()` function when rotate='Promax'
            z = z * np.sqrt(h2)

        rotation_mtx = np.dot(rotation_mtx, coef)

        coef_inv = np.linalg.inv(coef)
        phi = np.dot(coef_inv, coef_inv.T)

        # convert loadings matrix to data frame
        loadings = z.copy()
        return loadings, rotation_mtx, phi

    def fit(self, X: pd.DataFrame, y=None):
        """
        Computes the factor rotation,
        and returns the new loading matrix.

        Parameters
        ----------
        X : array-like
            The factor loading matrix (n_features, n_factors)
        y : Ignored

        Returns
        -------
        loadings_ : numpy array, shape (n_features, n_factors)
            The loadings matrix
            (n_features, n_factors)

        Raises
        ------
        ValueError
            If the `method` is not in the list of
            acceptable methods.

        Example
        -------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer, Rotator
        >>> df_features = pd.read_csv('test02.csv')
        >>> fa = FactorAnalyzer(rotation=None)
        >>> fa.fit(df_features)
        >>> rotator = Rotator()
        >>> rotator.fit_transform(fa.loadings_)
        array([[-0.07693215,  0.04499572,  0.76211208],
               [ 0.01842035,  0.05757874,  0.01297908],
               [ 0.06067925,  0.70692662, -0.03311798],
               [ 0.11314343,  0.84525117, -0.03407129],
               [ 0.15307233,  0.5553474 , -0.00121802],
               [ 0.77450832,  0.1474666 ,  0.20118338],
               [ 0.7063001 ,  0.17229555, -0.30093981],
               [ 0.83990851,  0.15058874, -0.06182469],
               [ 0.76620579,  0.1045194 , -0.22649615],
               [ 0.81372945,  0.20915845,  0.07479506]])
        """
        # default phi to None
        # it will only be calculated
        # for oblique rotations
        phi = None
        method = self.method.lower()
        if method == "varimax":
            (new_loadings, new_rotation_mtx) = self._varimax(X)

        elif method == "promax":
            (new_loadings, new_rotation_mtx, phi) = self._promax(X)

        elif method in OBLIQUE_ROTATIONS:
            (new_loadings, new_rotation_mtx, phi) = self._oblique(X, method)

        elif method in ORTHOGONAL_ROTATIONS:
            (new_loadings, new_rotation_mtx) = self._orthogonal(X, method)

        else:
            rot_str = ", ".join(POSSIBLE_ROTATIONS)
            raise ValueError(
                "The value for `method` must be one of the " f"following: {rot_str}."
            )

        (self.loadings_, self.rotation_, self.phi_) = (
            new_loadings,
            new_rotation_mtx,
            phi,
        )
        return self

    def transform(self, loadings: pd.DataFrame) -> pd.DataFrame:
        """Rotate loadings based on fit parameters."""
        if self.method == "varimax":
            return self._varimax_transform(loadings)
        if self.method == "promax":
            raise NotImplementedError("This method is not yet implemented.")
        if self.method in OBLIQUE_ROTATIONS:
            return self._oblique_transform(loadings)
        if self.method in ORTHOGONAL_ROTATIONS:
            return self._orthogonal_transform(loadings)
        rot_str = ", ".join(POSSIBLE_ROTATIONS)
        raise ValueError(
            "The value for `method` must be one of the " f"following: {rot_str}."
        )
