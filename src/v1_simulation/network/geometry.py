import numpy as np


class SheetGeometry:
    """A 2D regular grid sheet of neurons embedded in 3D space.

    Represents a flat layer of neurons arranged in a square grid with physical
    coordinates, supporting periodic boundary conditions and distance matrix calculations.
    """

    def __init__(self, n_side, region_size, z_pos):
        """Initializes the SheetGeometry.

        Args:
            n_side: Number of neurons along one side of the square sheet grid (total N = n_side^2).
            region_size: The physical size (width and height) of the sheet.
            z_pos: The 3D z-coordinate (depth) position of this layer.
        """
        self.n_side = self._validate_positive_int(n_side, "n_side")
        self.N = self.n_side * self.n_side
        self.region_size = self._validate_positive_float(region_size, "region_size")
        self.z_pos = float(z_pos)

        self.coords = self._generate_grid_positions()

    @staticmethod
    def _validate_positive_int(value, name):
        """Validates that a value is a positive integer.

        Args:
            value: Value to validate.
            name: Parameter name for error messages.

        Returns:
            The validated integer.

        Raises:
            ValueError: If the value is not positive.
        """
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
        return value

    @staticmethod
    def _validate_positive_float(value, name):
        """Validates that a value is a positive float.

        Args:
            value: Value to validate.
            name: Parameter name for error messages.

        Returns:
            The validated float.

        Raises:
            ValueError: If the value is not positive.
        """
        value = float(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
        return value

    def _generate_grid_positions(self):
        """Generates the 2D grid coordinates for neurons on the sheet.

        Returns:
            A numpy array of shape (N, 2) containing (x, y) coordinates of each neuron.
        """
        spacing = self.region_size / self.n_side
        axis = (np.arange(self.n_side, dtype=float) + 0.5) * spacing
        axis -= self.region_size / 2.0
        x, y = np.meshgrid(axis, axis, indexing="xy")
        return np.column_stack((x.ravel(), y.ravel()))

    @staticmethod
    def _periodic_delta(delta, box_size):
        """Applies periodic boundary wrapping to coordinate differences.

        Args:
            delta: Array of differences in coordinates.
            box_size: The dimensions of the periodic box.

        Returns:
            Wrapped coordinate differences.
        """
        return (delta + box_size / 2.0) % box_size - box_size / 2.0

    def get_distance_matrix(self, periodic=True):
        """Calculates the pairwise Euclidean distance matrix for neurons within this layer.

        Args:
            periodic: If True, uses periodic boundary conditions (toroidal wrapping)
              for calculating 2D lateral distances.

        Returns:
            A 2D numpy array of shape (N, N) containing pairwise distances.
        """
        delta = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        if periodic:
            delta = self._periodic_delta(delta, self.region_size)
        return np.linalg.norm(delta, axis=2)

    def get_distance_to(self, other_layer, periodic=True):
        """Calculates the Euclidean distance matrix from this layer to another layer.

        Args:
            other_layer: The SheetGeometry instance representing the target layer.
            periodic: If True, uses periodic boundary conditions. Requires both
              layers to have matching region_sizes.

        Returns:
            A 2D numpy array of shape (self.N, other_layer.N) containing pairwise distances.

        Raises:
            ValueError: If periodic is True and region_sizes do not match.
        """
        if periodic and not np.isclose(self.region_size, other_layer.region_size):
            raise ValueError(
                "Periodic cross-layer distance requires matching region_size: "
                f"{self.region_size} vs {other_layer.region_size}."
            )

        delta_2d = self.coords[:, np.newaxis, :] - other_layer.coords[np.newaxis, :, :]
        if periodic:
            delta_2d = self._periodic_delta(delta_2d, self.region_size)

        dist_2d_sq = np.sum(delta_2d**2, axis=2)
        z_diff = self.z_pos - other_layer.z_pos
        return np.sqrt(dist_2d_sq + z_diff**2)


class L4(SheetGeometry):
    """Layer 4 (L4) geometry and neuron properties.

    Represents the input layer 4 where neurons can be tuned or untuned to visual
    stimuli, and assigns preferred orientation directions.
    """

    def __init__(self, cfg_l4, exp_data, rng=None):
        """Initializes L4 layer.

        Args:
            cfg_l4: Configuration object containing L4 layer settings.
            exp_data: Experimental data or constraints (containing NT_X or pT_X).
            rng: Random number generator instance (defaults to np.random).
        """
        super().__init__(cfg_l4.n_side, cfg_l4.region_size, cfg_l4.z_pos)
        self.cfg = cfg_l4
        self.exp_data = exp_data
        self.rng = np.random if rng is None else rng
        self._set_neurons()

    @staticmethod
    def _bounded_count(count, total, name):
        """Ensures that a count lies within the valid range [0, total].

        Args:
            count: Number to check.
            total: Maximum allowed value.
            name: Parameter name for error messages.

        Returns:
            The validated integer count.

        Raises:
            ValueError: If count is out of bounds.
        """
        count = int(count)
        if count < 0 or count > total:
            raise ValueError(f"{name} must be between 0 and {total}, got {count}.")
        return count

    def _tuned_count(self):
        """Determines the number of tuned neurons in L4 based on experimental data.

        Returns:
            An integer count of tuned neurons.
        """
        if self.cfg.l4.all_tuned:
            return self.N

        if hasattr(self.exp_data, "NT_X"):
            return self._bounded_count(self.exp_data.NT_X, self.N, "NT_X")

        p_tuned = float(self.exp_data.pT_X)
        return self._bounded_count(round(self.N * p_tuned), self.N, "NT_X")

    def _set_neurons(self):
        """Sets neuron types (tuned vs untuned) and their preferred orientations."""
        n_tuned = self._tuned_count()

        self.tunings = np.full(self.N, "U", dtype="<U1")
        if n_tuned:
            tuned_idx = self.rng.choice(self.N, size=n_tuned, replace=False)
            self.tunings[tuned_idx] = "T"

        self.pref_dirs = np.full(self.N, np.nan, dtype=float)
        tuned_mask = self.tunings == "T"
        n_tuned_actual = int(np.sum(tuned_mask))
        if n_tuned_actual == 0:
            return

        theta_values = np.linspace(0.0, 2.0 * np.pi, self.cfg.N_theta, endpoint=False)
        pref_dirs = np.resize(theta_values, n_tuned_actual)
        self.rng.shuffle(pref_dirs)
        self.pref_dirs[tuned_mask] = pref_dirs


class L2_3(SheetGeometry):
    """Layer 2/3 (L2/3) geometry and neuron properties.

    Represents the cortical layer 2/3 containing excitatory (E) and inhibitory (I)
    neurons, arranged either randomly or uniformly on a grid.
    """

    def __init__(self, cfg_l23, exp_data, rng=None):
        """Initializes L2/3 layer.

        Args:
            cfg_l23: Configuration object containing L2/3 layer settings.
            exp_data: Experimental data containing neuron counts (e.g. N_I, N_E, l2_3_n_side).
            rng: Random number generator instance (defaults to np.random).
        """
        super().__init__(exp_data.l2_3_n_side, cfg_l23.region_size, cfg_l23.z_pos)
        self.cfg = cfg_l23
        self.exp_data = exp_data
        self.rng = np.random if rng is None else rng
        self._set_neurons()

    @staticmethod
    def _bounded_count(count, total, name):
        """Ensures that a count lies within the valid range [0, total].

        Args:
            count: Number to check.
            total: Maximum allowed value.
            name: Parameter name for error messages.

        Returns:
            The validated integer count.

        Raises:
            ValueError: If count is out of bounds.
        """
        count = int(count)
        if count < 0 or count > total:
            raise ValueError(f"{name} must be between 0 and {total}, got {count}.")
        return count

    def _expected_counts(self):
        """Calculates expected counts of excitatory and inhibitory neurons in L2/3.

        Returns:
            A tuple of (excitatory_count, inhibitory_count).

        Raises:
            ValueError: If N_E + N_I does not equal total layer size N.
        """
        n_i = self._bounded_count(self.exp_data.N_I, self.N, "N_I")
        n_e = self.N - n_i

        if hasattr(self.exp_data, "N_E") and int(self.exp_data.N_E) != n_e:
            raise ValueError(
                f"N_E + N_I must match layer size. "
                f"Got N_E={self.exp_data.N_E}, N_I={n_i}, layer N={self.N}."
            )

        return n_e, n_i

    def _uniform_grid_indices(self, count):
        """Generates indices for inhibitory neurons to place them as uniformly as possible on the grid.

        Args:
            count: The target number of inhibitory neurons to select.

        Returns:
            A 1D numpy array of integer indices representing selected neuron positions.
        """
        count = self._bounded_count(count, self.N, "N_I")
        if count == 0:
            return np.array([], dtype=int)
        if count == self.N:
            return np.arange(self.N, dtype=int)

        n_cols = min(self.n_side, int(np.ceil(np.sqrt(count))))
        n_rows = min(self.n_side, int(np.ceil(count / n_cols)))

        while n_rows * n_cols < count:
            if n_cols < self.n_side:
                n_cols += 1
            elif n_rows < self.n_side:
                n_rows += 1
            else:
                break

        rows = np.rint(np.linspace(0, self.n_side - 1, n_rows)).astype(int)
        cols = np.rint(np.linspace(0, self.n_side - 1, n_cols)).astype(int)

        candidates = np.array(
            [r * self.n_side + c for r in rows for c in cols],
            dtype=int,
        )
        candidates = np.unique(candidates)

        if candidates.size > count:
            keep = np.linspace(0, candidates.size - 1, count, dtype=int)
            candidates = candidates[keep]

        if candidates.size < count:
            all_indices = np.arange(self.N, dtype=int)
            remaining = np.setdiff1d(all_indices, candidates, assume_unique=True)
            extra = remaining[np.linspace(0, remaining.size - 1, count - candidates.size, dtype=int)]
            candidates = np.concatenate([candidates, extra])

        return candidates

    def _set_neurons(self):
        """Sets L2/3 neuron types (E/I) and assigns their grid positions."""
        _, n_i = self._expected_counts()

        self.types = np.full(self.N, "E", dtype="<U1")
        if self.cfg.random_I:
            inhibitory_idx = self.rng.choice(self.N, size=n_i, replace=False)
        else:
            inhibitory_idx = self._uniform_grid_indices(n_i)

        self.types[inhibitory_idx] = "I"