import AbstractBasis
import numpy as np

class FunctionCombiner:
    def __init__(self, basis: AbstractBasis, coefficients: np.ndarray = None):
        """
        :param basis: an object implemented AbstractBasis interface
        :param coefficients: a list with coefficients
        """
        self.basis = basis

        # Set coefficients to be 0s if they are not specified
        if coefficients is None:
            self.coefficients = np.zeros(self.basis.num_basis())
        else:
            self.set_coefficients(coefficients)

    def set_coefficients(self, coeffs: np.ndarray):
        """
        Set coefficients
        """
        coeffs = np.array(coeffs, dtype=float)
        if coeffs.shape[1] != self.basis.dim_basis():
            raise ValueError(
                f"Length of coeffs ({coeffs.shape[1]}) != dim_basis ({self.basis.dim_basis()})"
            )
        self.coefficients = coeffs

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the value of this linear combination: f(x) = \sum_i c_i * basis(x)
        :param x: a numpy array, vector like, points to be evaluated
        :return: A numpy ndarray
        """
        return self.coefficients @ self.basis.evaluate_basis(x).transpose()

    def plotHeatmap(self) -> None:
        # TODO: plot the heatmap using H
        return None

