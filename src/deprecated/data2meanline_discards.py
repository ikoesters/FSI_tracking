    
    def rotate_points(self, points: np.array, inclination_in_arc: float) -> np.array:
        rot_mat = ut.rot_matrix(inclination_in_arc)
        rotated = np.zeros_like(points)
        for idx, (rot, pts) in enumerate(zip(rot_mat, points)):
            rotated[idx] = rot @ pts
        return rotated

    def improve_midline(self, midline: np.array, outline: np.array) -> np.array:
        arc = np.arctan(np.gradient(midline[:, 1], midline[:, 0]))
        arc_resampled = np.interp(outline[:, 0], midline[:, 0], arc)

        outline_rotated = self.rotate_points(outline, arc_resampled)
        midline = ut.fit(
            outline_rotated,
            3,
            np.linspace(outline_rotated[:, 0].min(), outline_rotated[:, 0].max(), 1000),
        )
        midline_backrotated = self.rotate_points(midline, -arc_resampled)
        return midline_backrotated



    def split_outline(self, outline: np.array) -> np.array:
        outline = outline[outline[:, 0].argsort()]  # Sort for x
        lower_side = np.zeros((0, 2))
        upper_side = np.zeros((0, 2))
        for k in range(outline[0, 0], outline[-1, 0]):
            # grab all entries where x=i
            x_equal_k = outline[(outline[:, 0] == k)]
            mean_at_x = np.mean(x_equal_k, axis=0)

            if len(x_equal_k) == 1:  # if only one point
                continue

            lower_than_mean = x_equal_k[(x_equal_k[:, 1] < mean_at_x[1])]
            lower_than_mean = np.mean(lower_than_mean, axis=0)
            lower_side = np.vstack((lower_side, lower_than_mean))

            higher_than_mean = x_equal_k[(x_equal_k[:, 1] > mean_at_x[1])]
            higher_than_mean = np.mean(higher_than_mean, axis=0)
            upper_side = np.vstack((upper_side, higher_than_mean))
        return lower_side, upper_side
    
    def compute_midlines(self) -> np.array:
        self.getData.act_frame = 0
        midlines = np.zeros((0, self.sample_points, 2))
        print("Compute Midlines")
        for framenb in trange(0, self.getData.stop_frame, self.sum_nb):
            midline = self.compute_midline(framenb=framenb)
            midline = np.expand_dims(midline, axis=0)
            midlines = np.vstack((midlines, midline))
        return midlines