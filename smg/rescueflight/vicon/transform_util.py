import numpy as np

from typing import Dict, Optional

from smg.utility import GeometryUtil


class TransformUtil:
    """Utility functions to calculate the transformations between the various different spaces."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def try_calculate_aruco_from_world(marker_positions: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Try to calculate the transformation from some world space to the space associated with an ArUco marker
        by making use of the known world-space positions of the ArUco marker's corners.

        :param marker_positions:    The world-space positions of the ArUco marker's corners.
        :return:                    The transformation from world space to ArUco space, if possible, or None otherwise.
        """
        # If the positions of all of the ArUco marker's corners are known, estimate the transformation.
        if all(key in marker_positions for key in ["0_0", "0_1", "0_2", "0_3"]):
            p: np.ndarray = np.column_stack([
                marker_positions["0_0"],
                marker_positions["0_1"],
                marker_positions["0_2"],
                marker_positions["0_3"]
            ])

            offset: float = 0.0705  # 7.05cm (half the width of the printed marker)
            q: np.ndarray = np.array([
                [-offset, -offset, 0],
                [offset, -offset, 0],
                [offset, offset, 0],
                [-offset, offset, 0]
            ]).transpose()

            return GeometryUtil.estimate_rigid_transform(p, q)

        # Otherwise, return None.
        else:
            return None

    @staticmethod
    def try_calculate_vicon_from_gt(gt_marker_positions: Dict[str, np.ndarray],
                                    vicon_marker_positions: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Try to calculate the transformation from ground-truth space to Vicon space.

        .. note::
            This approach is based on using an ArUco marker with Vicon markers attached to its corners.
            The ground-truth space positions of the ArUco marker corners are estimated using SemanticPaint,
            and the Vicon-space positions of the ArUco marker corners are obtained directly from the Vicon
            markers. The correspondences between the two can be used to estimated the transform.

        :param gt_marker_positions:     The positions of the ArUco marker corners in ground-truth space.
        :param vicon_marker_positions:  The Vicon space positions of the all of the Vicon markers for the ArUco
                                        marker subject that can currently be seen by the Vicon.
        :return:                        The transformation from ground-truth space to Vicon space, if possible,
                                        or None otherwise.
        """
        # If all of the ArUco marker corners can be seen, estimate the world space to Vicon space transformation.
        if all(key in vicon_marker_positions for key in ["0_0", "0_1", "0_2", "0_3"]):
            p: np.ndarray = np.column_stack([
                gt_marker_positions["0_0"],
                gt_marker_positions["0_1"],
                gt_marker_positions["0_2"],
                gt_marker_positions["0_3"]
            ])

            q: np.ndarray = np.column_stack([
                vicon_marker_positions["0_0"],
                vicon_marker_positions["0_1"],
                vicon_marker_positions["0_2"],
                vicon_marker_positions["0_3"]
            ])

            return GeometryUtil.estimate_rigid_transform(p, q)

        # Otherwise, return None.
        else:
            return None
