import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.image import imsave
from matplotlib.colors import LightSource, Normalize
import seaborn as sns
import imageio
from PIL import Image
from pathlib import Path
import cortex as cx


def zoom_to_roi(subject="fsaverage", roi="V2", hem="left", margin=30.0):
    roi_verts = cx.get_roi_verts(subject, roi)[roi]
    roi_map = cx.Vertex.empty(subject)
    roi_map.data[roi_verts] = 1

    (lflatpts, lpolys), (rflatpts, rpolys) = cx.db.get_surf(subject, "flat", nudge=True)
    sel_pts = dict(left=lflatpts, right=rflatpts)[hem]
    roi_pts = sel_pts[np.nonzero(getattr(roi_map, hem))[0], :2]

    xmin, ymin = roi_pts.min(0) - margin
    xmax, ymax = roi_pts.max(0) + margin
    plt.axis([xmin, xmax, ymin, ymax])


def con_weighted_dispcx(subject, data, rsq, vmin, vmax, cmap="seismic"):
    curv = cx.db.get_surfinfo(subject)
    # Adjust curvature contrast / color. Alternately, you could work
    # with curv.data, maybe threshold it, and apply a color map.
    curv.vmin = -0.1
    curv.vmax = 0.1
    curv.cmap = "gray"
    curv.data = curv.data * 0.75 + 0.1
    # curv = cortex.Vertex(curv.data, subject, vmin=-1,vmax=1,cmap='gray')
    # Create some display data

    # normalize the range of rsq to 0 to 1
    if rsq.max() - rsq.min() > 0:
        rsq = (rsq - rsq.min()) / (rsq.max() - rsq.min() - 0.2)
    else:
        pass

    vx = cx.Vertex(
        data,
        subject,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    # Map to RGB
    vx_rgb = np.vstack([vx.raw.red.data, vx.raw.green.data, vx.raw.blue.data])
    curv_rgb = np.vstack([curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data])

    alpha = rsq
    alpha = alpha.astype(np.float32)

    # Alpha mask
    display_data = vx_rgb * alpha + curv_rgb * (1 - alpha)
    # fake_curv_rgb = np.zeros_like(curv_rgb)
    # display_data = vx_rgb * alpha + fake_curv_rgb * (1 - alpha)
    # display_data /= 255
    return display_data


def display_webgl_rsqw(cx_subject, data, rsq, vmin, vmax, deriv_dir=None, port=12001):
    display_viol_ratio = con_weighted_dispcx(
        cx_subject, data, rsq, vmin=vmin, vmax=vmax
    )
    cx_web_rsqw = cx.VertexRGB(
        *display_viol_ratio,
        cx_subject,
    )
    if deriv_dir is not None:
        cx.webgl.make_static(
            deriv_dir / "webgl", cx_web_rsqw
        )  # This step will save the webgl files
    handle = cx.webgl.show(data=cx_web_rsqw, recache=True, port=port)
    return handle


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def simple_colorbar(vmin, vmax, cmap_name, ori, param_name):
    # from Marco's prftools https://github.com/VU-Cog-Sci/prfpytools/blob/new-branch/prfpytools/postproc_utils.py
    if ori == "horizontal":
        fig, ax = plt.subplots(figsize=(8, 1))
        fig.subplots_adjust(bottom=0.5)
    elif ori == "vertical":
        fig, ax = plt.subplots(figsize=(3, 8))
        fig.subplots_adjust(right=0.5)
    elif ori == "polar":
        fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={"projection": "polar"})

    if isinstance(cmap_name, str):
        if cmap_name == "hsvx2":
            top = cm.get_cmap("hsv", 256)
            bottom = cm.get_cmap("hsv", 256)

            newcolors = np.vstack(
                (top(np.linspace(0, 1, 256)), bottom(np.linspace(0, 1, 256)))
            )
            cmap = colors.ListedColormap(newcolors, name="hsvx2")

        else:
            cmap = cm.get_cmap(cmap_name, 256)
    elif isinstance(cmap_name, colors.ListedColormap):
        cmap = cmap_name

    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    if ori == "polar":
        if "Polar" in param_name:
            t = np.linspace(-np.pi, np.pi, 800, endpoint=True)
            r = [0, 1]
            rg, tg = np.meshgrid(r, t)
            ax.pcolormesh(
                t,
                r,
                tg.T,
                norm=norm,
                cmap=cmap,
                edgecolors="none",
                shading="gouraud",
                linewidth=0,
            )
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(-1)
            ax.grid(False)
            ax.spines["polar"].set_visible(False)
        elif "Ecc" in param_name:
            n = 200
            t = np.linspace(0, 2 * np.pi, n)
            r = np.linspace(0, 1, n)
            rg, tg = np.meshgrid(r, t)
            c = tg
            ax.pcolormesh(
                t,
                r,
                c,
                norm=colors.Normalize(0, 2 * np.pi),
                cmap=cmap,
                edgecolors="none",
                shading="gouraud",
                linewidth=0,
            )
            ax.tick_params(pad=1, labelsize=15)
            ax.spines["polar"].set_visible(False)
            box = ax.get_position()
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(False)
            # axl = fig.add_axes([0.97*box.xmin,0.5*(box.ymin+box.ymax), box.width/600,box.height*0.5])
            # axl.spines['top'].set_visible(False)
            # axl.spines['right'].set_visible(False)
            # axl.spines['bottom'].set_visible(False)
            # # for ax in fig.axes:
            # #     for spine in ax.spines.values():
            # #         spine.set_visible(False)
            # axl.yaxis.set_ticks_position('left')
            # axl.xaxis.set_ticks_position('none')
            # axl.set_xticklabels([])
            # axl.set_yticks([vmin, np.mean([vmin, vmax]), vmax])
            # axl.set_yticklabels([f"{vmin:.1f}",f"{(vmin+vmax)/2:.1f}",f"{vmax:.1f}"],size = 'x-large')
            # #axl.set_ylabel('$dva$\t\t', rotation=0, size='x-large')
            # axl.yaxis.set_label_coords(box.xmax+30,0.4)
            # axl.patch.set_alpha(0.5)
    else:
        fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation=ori,
            label=param_name,
        )

    return fig


# Create a function to desaturate colors
def desaturate_cmap(cmap, saturation=0.5, lightness=1.0):
    """
    Example usage
    original_cmap = plt.cm.viridis  # Original colormap
    desaturated_cmap = desaturate_cmap(original_cmap, saturation=0.5)
    data = np.random.rand(10, 10)
    plt.imshow(data, cmap=desaturated_cmap)
    plt.colorbar()
    plt.show()
    """

    def desaturate_color(color):
        color = colors.rgb_to_hsv(colors.to_rgb(color))
        color[1] = color[1] * saturation  # Modify the saturation (index 1 in HSV)
        color[2] = color[2] * lightness  # Modify the lightness (index 2 in HSV)
        return colors.hsv_to_rgb(color)

    return colors.ListedColormap(
        [desaturate_color(c) for c in cmap(np.linspace(0, 1, cmap.N))]
    )


def desaturate_palette(cmap, saturation=0.5, lightness=1.0):
    # Get the desaturated colormap
    # cmap: colormap to desaturate, should be a matplotlib.colors.LinearSegmentedColormap
    # saturation: float, saturation factor (0.0 to 1.0)
    # lightness: float, lightness factor (0.0 to 1.0)
    # usage example:
    # example 1:
    # desaturated_cmap = desaturate_palette(plt.cm.viridis, saturation=0.5, lightness=1.0)
    # example 2:
    # desaturated_cmap = desaturate_palette('viridis', saturation=0.5, lightness=1.0)
    # example 3:
    # from matplotlib.colors import LinearSegmentedColormap
    # colors = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)]
    # customparamPalette = LinearSegmentedColormap.from_list("custom_cmap", colors, N=len(colors))
    # customparamPalette_desaturated = desaturate_palette(customparamPalette, saturation=1.0, lightness=0.6)
    de_cmap = desaturate_cmap(cmap, saturation, lightness)
    n_colors = len(de_cmap.colors)
    pre_color_palette = [de_cmap(i / n_colors) for i in range(n_colors)]
    palette_desaturated_cmap = sns.color_palette(pre_color_palette)
    return palette_desaturated_cmap


def reorder_cmap(cmap_name, values, num_bars):
    cmap = plt.colormaps.get_cmap(cmap_name)
    norm = plt.Normalize(values.min(), values.max())
    colors = cmap(norm(values))
    return list(colors)


def sample_cmap(cmap, num_bars):
    # If cmap is a string, get the colormap using the string name
    if isinstance(cmap, str):
        cmap = plt.colormaps.get_cmap(cmap)
        colors = [cmap(value) for value in np.linspace(0, 1, num_bars)]
    # If cmap is a callable function (like a matplotlib colormap)
    elif callable(cmap):
        colors = [cmap(value) for value in np.linspace(0, 1, num_bars)]
    # If cmap is a list or non-callable, assume it's already a list of colors
    else:
        # Ensure num_bars does not exceed the number of available colors
        if num_bars > len(cmap):
            raise ValueError(
                "Number of bars exceeds the number of colors in the colormap"
            )
        colors = [cmap[i] for i in np.linspace(0, len(cmap) - 1, num_bars).astype(int)]
    return list(colors)


def plot_3D_sphere(ax, radius, x, y, z, color="b", zorder=1):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    u = np.linspace(0, 2 * np.pi, 100)  # Azimuthal angle
    v = np.linspace(0, np.pi, 100)  # Polar angle
    # Generate the coordinates for the sphere and zoom the radius according to the xlim, ylim, and zlim
    x = radius * np.outer(np.cos(u), np.sin(v)) * (xlim[1] - xlim[0]) / 100 + x
    y = radius * np.outer(np.sin(u), np.sin(v)) * (ylim[1] - ylim[0]) / 100 + y
    z = (
        radius * np.outer(np.ones(np.size(u)), np.cos(v)) * (zlim[1] - zlim[0]) / 100
        + z
    )

    # Use a color map to simulate material
    ax.plot_surface(
        x,
        y,
        z,
        rstride=15,
        cstride=15,
        color=color,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
        shade=True,
        zorder=zorder,
        # lightsource=LightSource(azdeg=315, altdeg=45),
    )

    # Enhance lighting with shading (pseudo-lighting)
    ax.plot_surface(
        x, y, z, color="white", alpha=0.25, linewidth=0, antialiased=True, zorder=zorder
    )

    # Set the aspect ratio to be equal, so the sphere looks correct
    ax.set_box_aspect([1, 1, 1])

    return ax


def create_retinotopy_2x_colormaps():
    hue, alpha = np.meshgrid(
        np.linspace(0, 1, 256, endpoint=False), 1 - np.linspace(0, 1, 1)
    )
    print(hue.shape)
    hsv = np.zeros(list(hue.shape) + [3])
    print(hsv.shape)
    # convert angles to colors, using correlations as weights
    hsv[..., 0] = hue  # angs_discrete  # angs_n
    # np.sqrt(rsq) #np.ones_like(rsq)  # np.sqrt(rsq)
    hsv[..., 1] = np.ones_like(alpha)
    # np.nan_to_num(rsq ** -3) # np.ones_like(rsq)#n
    hsv[..., 2] = np.ones_like(alpha)
    hsv_2x = np.hstack([hsv, hsv])
    hsv_2x = hsv_2x[:, 0::2, :]
    rgb_2x = colors.hsv_to_rgb(hsv_2x)
    plt.imshow(rgb_2x)
    hsv_fn = str(
        Path(cx.database.default_filestore).absolute().parent
        / "colormaps"
        / "hsv_2x.png"
    )
    rgb_2x = Image.fromarray((rgb_2x * 255).astype(np.uint8))
    print(hsv_fn)
    imageio.imwrite(hsv_fn, rgb_2x)


def create_alpha_colormaps(cmap_name, reverse=False):
    n = 256  # Size of the image (resolution)
    x = np.linspace(0, 1, n)  # Linear space for the colormap
    y = np.linspace(1, 0, n)  # Linear space for the alpha gradient

    # Normalize and apply the colormap
    norm = Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap(cmap_name)

    # Create a meshgrid to combine color and alpha channels
    X, Y = np.meshgrid(x, y)

    # Get color values from the colormap
    colors = cmap(norm(X))

    # Add an alpha gradient in the second dimension
    alpha_gradient = Y  # Alpha should be a 2D array matching X, Y
    if reverse:
        # horizontally flip the figure
        colors = np.fliplr(colors)

    colors[:, :, 3] = alpha_gradient  # Assign the alpha gradient to the 4th channel

    # Plot and display the result
    fig, ax = plt.subplots()

    # Plot and display the result
    img = ax.imshow(colors, origin="upper", aspect="auto")
    ax.axis("off")

    if reverse:
        fn_cmap_name = f"{cmap_name}_alpha_r.png"
    else:
        fn_cmap_name = f"{cmap_name}_alpha.png"

    fn = str(
        Path(cx.database.default_filestore).absolute().parent
        / "colormaps"
        / fn_cmap_name
    )
    print(fn)
    # Save the colormap to a file
    imsave(fn, colors)


def cm_to_inch(*dims):
    return tuple(dim / 2.54 for dim in dims)
