import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt


def visualize_point_cloud(data, labels, classes, idx):
    x, y, z = data[idx].squeeze().numpy().T
    print(labels.shape)
    label = labels[idx].item()
    key = list(classes.keys())[list(classes.values()).index(label)]
    data_obj = [
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=3, color=z, colorscale="Viridis", opacity=0.8),
            # marker=dict(size=4, color=z, colorscale="Viridis", opacity=0.8),
        )
    ]
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8  # x, y, z are the coordinates of the camera
    frames = []

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(
            dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze)))))
        )
    fig = go.Figure(
        data=data_obj,
        layout=go.Layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1,
                    x=0.8,
                    xanchor="left",
                    yanchor="bottom",
                    pad=dict(t=45, r=10),
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=50, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        )
                    ],
                )
            ]
        ),
        frames=frames,
    )
    # add title
    fig.update_layout(title_text=f"{key} - 3D Point Cloud Rotation")

    fig.show()

def visualize_transformation(point_cloud, transformation=None):
    if transformation is None:
        raise ValueError("transformation must be specified.")
    transformed_points = transformation(point_cloud)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": "3d"})
    axs[0].scatter(
        point_cloud[..., 0], point_cloud[..., 1], point_cloud[..., 2], c="r", marker="o"
    )
    axs[0].set_title("Original Point Cloud")

    axs[1].scatter(
        transformed_points[..., 0],
        transformed_points[..., 1],
        transformed_points[..., 2],
        c="b",
        marker="o",
    )
    axs[1].set_title("Transformed Point Cloud")
    fig.tight_layout()
    fig.show()