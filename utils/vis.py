from math import ceil, prod, sqrt
import os

import mitsuba as mi
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import draw_keypoints
import torchvision.transforms.functional as torchvision_F
import matplotlib
import matplotlib.pyplot as plt
import PIL
import PIL.ImageDraw
import imageio

from .camera import get_keypoint_projections, get_vis_intr


def tb_image(
    opt, 
    tb, 
    step, 
    group, 
    name, 
    images, 
    masks=None, 
    kp=None, 
    extr=None, 
    intr=None,
    pix_scale=None,
    pix_offset=None,
    from_range=(0, 1), 
    cmap="gray"
):
    images = preprocess_vis_image(images, masks=masks, from_range=from_range, cmap=cmap, size=opt.sizes.vis)
    B, num_views = images.shape[:2]
    ex_ins = min(opt.tb.num_instances, B)
    ex_view = min(opt.tb.num_views, num_views)
    images = images[:ex_ins, :ex_view].flatten(0, 1)
    num_cols = ceil(sqrt(ex_ins * ex_view))
    if extr is not None:
        R = extr[:ex_ins, :ex_view, :3, :3].flatten(0, 1)
        images = torch.stack([draw_pose(opt, image, r, size=20, width=2) for image, r in zip(images, R)], dim=0)
    if kp is not None:
        assert extr is not None and intr is not None
        if kp.dim() < 4:
            kp = kp[:ex_ins, None]
        else:
            kp = kp[:ex_ins, :ex_view]
        intr = intr[:ex_ins, :ex_view]
        extr = extr[:ex_ins, :ex_view]
        if pix_scale is not None:
            pix_scale = pix_scale[:ex_ins, :ex_view]
        if pix_offset is not None:
            pix_offset = pix_offset[:ex_ins, :ex_view]
        intr = get_vis_intr(intr, opt.sizes.vis, opt.sizes.image)
        kp_pix, kp_depth = get_keypoint_projections(kp, extr, intr, pix_scale, pix_offset)
        kp_pix, kp_depth = kp_pix.flatten(0, 1), kp_depth.flatten(0, 1)
        images = torch.stack([draw_keypoints(opt, image, kp_pix[i], kp_depth[i]) for i, image in enumerate(images)], dim=0)
    image_grid = torchvision.utils.make_grid(images[:, :3], nrow=num_cols, pad_value=1.)
    if images.shape[1]==4:
        mask_grid = torchvision.utils.make_grid(images[:, 3:], nrow=num_cols, pad_value=1.)[:1]
        image_grid = torch.cat([image_grid, mask_grid], dim=0)
    tag = "{0}/{1}".format(group, name)
    tb.add_image(tag, image_grid, step)

def preprocess_vis_image(images, masks=None, from_range=(0, 1), cmap="gray", size=None):
    min, max = from_range
    images = (images-min)/(max-min)
    if masks is not None:
        images = torch.cat([images, masks], dim=-3)
    images = images.clamp(min=0, max=1).cpu()
    shape = images.shape[:-3]
    images = images.view(-1, *images.shape[-3:])
    if images.shape[1] == 1:
        images = get_heatmap(images[:, 0], cmap=cmap)
    elif images.shape[1] == 2:
        images = torch.cat((get_heatmap(images[:, 0], cmap=cmap), images[:, 1:]), dim=1)
    if size is not None:
        images = torchvision_F.resize(images,  size=size)
    images = images.view(*shape, *images.shape[-3:])
    return images

def weights_to_colors(weights, cmap_name: str = "hsv"):
    """
    Arguments:
        weights: [..., n]
    Returns:
        color_map: [..., 3]
    """
    n = weights.shape[-1]
    cmap = matplotlib.cm.get_cmap(cmap_name)
    colors = torch.from_numpy(cmap(np.linspace(0, 1, n, endpoint=False))).float().to(device=weights.device)
    color_map = torch.matmul(weights, colors[:, :3])
    return color_map

def tb_attn(opt, tb, step, group, kp, attn):
    """
    Arguments:
        opt: edict
        tb: Tensorboard
        step: training step
        group: group in tensorboard
        kp: [B, num_kp, 3]
        attn: [B, num_kp, num_emb]
    """
    B = kp.shape[0]
    ex_ins = min(opt.tb.num_instances, B)
    kp = kp[:ex_ins]
    attn = attn[:ex_ins]
    colors = torch.round(255 * weights_to_colors(attn))   # [ex_ins, num_kp, 3]
    tag = "{0}/attn".format(group)
    tb.add_mesh(
        tag, 
        kp, 
        colors,
        config_dict={
            "material": {
                "cls": "PointsMaterial",
                "size": opt.vis.attn.point_size
            }
        },
        global_step=step
    )

def tb_attn2(opt, tb, step, group, kp, attn):
    """
    Arguments:
        opt: edict
        tb: Tensorboard
        step: training step
        group: group in tensorboard
        kp: [B, num_kp, 3]
        attn: [B, num_kp, num_emb]
    """
    B = kp.shape[0]
    ex_ins = min(opt.tb.num_instances, B)
    kp = kp[:ex_ins]
    num_emb = attn.shape[-1]
    attn = attn[:ex_ins]
    colors = torch.round((255 * (1 - attn))).unsqueeze(-1).expand(-1, -1, -1, 3)
    for i in range(num_emb):
        tag = "{0}/attn_{1}".format(group, i)
        tb.add_mesh(
            tag, 
            kp, 
            colors[..., i, :],
            config_dict={
                "material": {
                    "cls": "PointsMaterial",
                    "size": opt.vis.attn.point_size
                }
            }, 
            global_step=step
        )

def tb_point_cloud(opt, tb, step, group, name, kp, extr, intr, pix_scale=None, pix_offset=None, gt=None):
    """
    Arguments:
        opt: edict
        tb: Tensorboard
        step: training step
        group: group in tensorboard
        kp: [B, num_sou, num_kp, 3]
        extr: [B, num_sou, num_tar, 4, 4]
        intr: [B, num_sou, num_tar, 4, 4]
        gt: [B, num_points, 3]
    """
    B, num_views = extr.shape[:2]
    ex_ins = min(opt.tb.num_instances, B)
    ex_view = min(opt.tb.num_views, num_views)
    num_images = ex_ins * ex_view
    num_cols = ceil(sqrt(num_images))
    images = torch.ones((num_images, 3, opt.sizes.vis, opt.sizes.vis), device=extr.device)
    intr = intr[:ex_ins, :ex_view]
    intr = get_vis_intr(intr, opt.sizes.vis, opt.sizes.image)
    extr = extr[:ex_ins, :ex_view]
    if pix_scale is not None:
        pix_scale = pix_scale[:ex_ins, :ex_view]
    if pix_offset is not None:
        pix_offset = pix_offset[:ex_ins, :ex_view]
    # Draw Poses
    R = extr[..., :3, :3].flatten(0, 1)
    images = torch.stack([draw_pose(opt, image, r, size=20, width=2) for image, r in zip(images, R)], dim=0)
    # Draw keypoints
    if kp.dim() < 4:
        kp_draw = kp[:ex_ins, None]
    else:
        kp_draw = kp[:ex_ins, :ex_view]
    kp_pix, kp_depth = get_keypoint_projections(kp_draw, extr, intr, pix_scale, pix_offset)
    kp_pix, kp_depth = kp_pix.flatten(0, 1), kp_depth.flatten(0, 1)
    images = torch.stack([draw_keypoints(opt, image, kp_pix[i], kp_depth[i], color=(0, 0, 255)) for i, image in enumerate(images)], dim=0)
    if gt is not None:
        # Draw ground truth point cloud in green
        gt = gt[:ex_ins, None]
        gt_pix, gt_depth = get_keypoint_projections(gt, extr, intr, pix_scale, pix_offset)
        gt_pix, gt_depth = gt_pix.flatten(0, 1), gt_depth.flatten(0, 1)
        images = torch.stack([draw_keypoints(opt, image, gt_pix[i], gt_depth[i], color=(0, 255, 0)) for i, image in enumerate(images)], dim=0)
    tag = f"{group}/{name}"
    image_grid = torchvision.utils.make_grid(images[:, :3], nrow=num_cols, pad_value=1.)
    tb.add_image(tag, image_grid, step)

def tb_point_cloud2(opt, tb, step, group, kp, gt):
    """
    Arguments:
        opt: edict
        tb: Tensorboard
        step: training step
        group: group in tensorboard
        kp: [B, num_kp, 3]
        gt: [B, num_points, 3]
    """
    B, num_kp = kp.shape[:2]
    ex_ins = min(opt.tb.num_instances, B)
    kp = kp[:ex_ins]
    pc_perm = torch.randperm(gt.shape[1], device=gt.device)[:opt.vis.gt_pc.samples]
    gt = gt[:ex_ins, pc_perm]
    cmap = matplotlib.cm.get_cmap(opt.vis.kp.cmap)
    colors = torch.from_numpy(cmap(np.linspace(0, 1, num_kp, endpoint=False))).float().expand(ex_ins, -1, -1)
    colors = torch.round(255 * colors)
    tb.add_mesh(
        f"{group}/gt", 
        gt,
        config_dict={
            "material": {
                "cls": "PointsMaterial",
                "size": opt.vis.attn.point_size
            }
        }, 
        global_step=step
    )
    tb.add_mesh(
        f"{group}/kp", 
        kp,
        colors,
        config_dict={
            "material": {
                "cls": "PointsMaterial",
                "size": opt.vis.attn.point_size
            }
        }, 
        global_step=step
    )


# radius 0.025 also looks good
XML_BALL_SEGMENT = \
"""
    <shape type="sphere">
        <float name="radius" value="0.015"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

XML_HEAD = \
"""
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="{},{},{}" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="{}"/>

        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="{}"/>
            <integer name="height" value="{}"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>

        <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
"""


XML_TAIL = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.2"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

def render_single_point_cloud(opt, out_path, kp, extr, intr):
    xml_segments = [XML_HEAD.format(*extr[..., :3, 3].tolist(), opt.sizes.dump, opt.sizes.dump)]
    for i in range(kp.shape[0]):
        xml_segments.append(XML_BALL_SEGMENT.format(*kp[i].tolist(), 0, 0, 1))
    xml_segments.append(XML_TAIL)
    xml_content = str.join('', xml_segments)
    with open('/tmp/mitsuba_scene.xml', 'w') as f:
        f.write(xml_content)
    mi.set_variant("scalar_rgb")
    scene = mi.load_file("/tmp/mitsuba_scene.xml")
    image = mi.render(scene) 
    mi.util.write_bitmap(out_path, image)


def render_single_point_cloud(opt, out_path, kp, extr, intr):
    # TODO compute fov from intr
    # fov = degrees(2 * np.arctan(resolution / (2 ** 0.5 * intr[0, 0])))
    fov = 53
    cam_loc = (-extr[..., :3, :3].transpose(-1, -2) @ extr[..., :3, 3:]).flatten()
    xml_segments = [XML_HEAD.format(*cam_loc.tolist(), fov, opt.sizes.render, opt.sizes.render)]
    for i in range(kp.shape[0]):
        xml_segments.append(XML_BALL_SEGMENT.format(*kp[i].tolist(), 0.46, 0.55, 0.68))
    xml_segments.append(XML_TAIL)
    xml_content = str.join('', xml_segments)
    with open('/tmp/mitsuba_scene.xml', 'w') as f:
        f.write(xml_content)
    mi.set_variant("scalar_rgb")
    scene = mi.load_file("/tmp/mitsuba_scene.xml")
    image = mi.render(scene) 
    mi.util.write_bitmap(out_path, image)


def dump_kp_pos_img(opt, obj_name, view_idx, kp, extr, intr):
    for i, o in enumerate(obj_name):
        d = os.path.join(opt.output_path, getattr(opt, "dump_dir", "dump"), o, "kp_pos_img")
        os.makedirs(d, exist_ok=True)
        for j, v in enumerate(view_idx[i]):
            f = os.path.join(d, str(v.item()) + ".png")
            render_single_point_cloud(opt, f, kp[i], extr[i, j], intr[i, j])


def dump_images(
    opt, 
    obj_name, 
    name, 
    view_idx, 
    images, 
    masks=None, 
    kp=None, 
    extr=None, 
    intr=None,
    pix_scale=None,
    pix_offset=None,
    from_range=(0, 1), 
    cmap="gray"
):
    B, num_views = images.shape[:2]
    images = preprocess_vis_image(images, masks=masks, from_range=from_range, cmap=cmap, size=opt.sizes.dump) # [B, num_views, 3, H, W]
    images = images.flatten(0, 1)
    if extr is not None:
        R = extr[:, :, :3, :3].flatten(0, 1)
        images = torch.stack([draw_pose(opt, image, r, size=20, width=2) for image, r in zip(images, R)], dim=0)
    if kp is not None:
        assert extr is not None and intr is not None
        kp = kp[:, None]
        intr = get_vis_intr(intr, opt.sizes.dump, opt.sizes.image)
        kp_pix, kp_depth = get_keypoint_projections(kp, extr, intr, pix_scale, pix_offset)
        kp_pix, kp_depth = kp_pix.flatten(0, 1), kp_depth.flatten(0, 1)
        images = torch.stack([draw_keypoints(opt, image, kp_pix[i], kp_depth[i]) for i, image in enumerate(images)],dim=0)
    images = images.view(B, num_views, *images.shape[-3:]).permute(0, 1, 3, 4, 2).cpu().numpy() # [B, num_views, H, W, 3]
    for i, o in enumerate(obj_name):
        d = os.path.join(opt.output_path, getattr(opt, "dump_dir", "dump"), o, name)
        os.makedirs(d, exist_ok=True)
        for j, v in enumerate(view_idx[i]):
            f = os.path.join(d, str(v.item()) + ".png")
            img_uint8 = (images[i, j]*255).astype(np.uint8)
            imageio.imsave(f, img_uint8)


def dump_emb_weights_as_alpha(opt, obj_name, view_idx, images, emb_weights):
    """
    view_idx: [B, num_views]
    images: [B, num_views, 3, H, W]
    emb_weights: [B, num_views, n_emb, H, W]
    """
    n_emb = emb_weights.shape[2]
    max_weight = torch.amax(emb_weights, dim=[-2, -1], keepdim=True)
    emb_weights = emb_weights / max_weight
    emb_weights.nan_to_num_()
    for i in range(n_emb):
        name = os.path.join("emb_weights_alpha", str(i))
        dump_images(opt, obj_name, name, view_idx, images, masks=emb_weights[:, :, [i]])


def dump_emb_weights_as_color(opt, obj_name, view_idx, emb_weights, attn_map):
    """
    view_idx: [B, num_views]
    emb_weights: [B, num_views, n_emb, H, W]
    attn_map: [B, n_kp, n_emb]
    """
    n_emb = attn_map.shape[2]
    attn_map = F.normalize(attn_map.transpose(1, 2), dim=-1)    # [B, n_emb, n_kp]
    pca = torch.pca_lowrank(attn_map, q=n_emb)[2]               # [B, n_kp, q]
    low_dim_attn = attn_map @ pca                               # [B, n_emb, q]
    low_dim_attn = torch.clamp(low_dim_attn, min=0)
    low_dim_attn = low_dim_attn / low_dim_attn.sum(dim=-1, keepdim=True)
    cmap = getattr(opt.vis.emb_weights, "color", "hsv") if hasattr(opt.vis, "emb_weights") else "hsv"
    emb_colors = weights_to_colors(low_dim_attn, cmap_name=cmap)    # [B, n_emb, 3]
    emb_img = (emb_weights.permute(0, 1, 3, 4, 2) @ emb_colors[:, None, None]).permute(0, 1, 4, 2, 3)   # [B, num_views, 3, H, W]
    dump_images(opt, obj_name, "emb_weights_color", view_idx, emb_img)


def dump_emb_weights_as_heat(opt, obj_name, view_idx, emb_weights):
    """
    view_idx: [B, num_views]
    emb_weights: [B, num_views, n_emb, H, W]
    """
    n_emb = emb_weights.shape[2]
    mask = emb_weights.sum(dim=2, keepdim=True)
    q = getattr(opt.vis.emb_weights, "quantile", 0.995) if hasattr(opt.vis, "emb_weights") else 0.995
    numel = prod(emb_weights.shape[-3:])
    upper_quantile = round((1-q) * numel)
    max_weight = torch.topk(emb_weights.flatten(-3), k=upper_quantile, dim=-1, largest=True).values[..., -1][..., None, None, None]
    # max_weight = torch.quantile(emb_weights.flatten(-3), q=q, dim=-1, keepdim=True)[..., None, None]     # torch.amax(emb_weights, dim=[-3, -2, -1], keepdim=True)    # [-3 in or out?]
    emb_weights = torch.clamp(emb_weights / max_weight, max=1.)
    emb_weights.nan_to_num_()
    cmap = getattr(opt.vis.emb_weights, "heat", "inferno") if hasattr(opt.vis, "emb_weights") else "inferno"
    for i in range(n_emb):
        name = os.path.join("emb_weights_heat", str(i))
        dump_images(opt, obj_name, name, view_idx, emb_weights[:, :, [i]], masks=mask, cmap=cmap)


def get_heatmap(gray, cmap): # [N, H, W]
    color = plt.get_cmap(cmap)(gray.numpy())
    color = torch.from_numpy(color[..., :3]).permute(0, 3, 1, 2).float() # [N, 3, H, W]
    return color


def dump_meshes(opt, idx, name, meshes):
    for i, mesh in zip(idx, meshes):
        fname = os.path.join(opt.output_path, getattr(opt, "dump_dir", "dump"), f"{i}_{name}.ply")
        mesh.export(fname)


@torch.no_grad()
def draw_pose(opt, image, rot_mtrx, size=15, width=1):
    mode = "RGBA" if image.shape[0]==4 else "RGB"
    image_pil = torchvision_F.to_pil_image(image.cpu()).convert("RGBA")
    draw_pil = PIL.Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(draw_pil)
    center = (size, size)
    endpoint = [(size+size*p[0], size+size*p[1]) for p in rot_mtrx.t()]
    draw.line([center, endpoint[0]], fill=(255, 0, 0), width=width)
    draw.line([center, endpoint[1]], fill=(0, 255, 0), width=width)
    draw.line([center, endpoint[2]], fill=(0, 0, 255), width=width)
    image_pil.alpha_composite(draw_pil)
    image_drawn = torchvision_F.to_tensor(image_pil.convert(mode))
    return image_drawn

"""
@torch.no_grad()
def draw_keypoints(opt, image, kp_pos, kp_depth):
    kp_pos = kp_pos.cpu()
    kp_depth = kp_depth.cpu()
    kp_num = kp_pos.size(0)
    mode = "RGBA" if image.shape[0]==4 else "RGB"
    image_pil = torchvision_F.to_pil_image(image.cpu()).convert("RGBA")
    fig, ax = plt.subplots()
    ax.imshow(image_pil)
    alpha = torch.clamp_min(kp_depth, min=opt.vis.kp.min_opacity)
    ax.scatter(kp_pos[..., 0], kp_pos[..., 1], s=opt.vis.kp.size, c=np.linspace(0, 1, kp_num, endpoint=False), cmap=opt.vis.kp.cmap, alpha=alpha)
    ax.axis('tight'); ax.axis('off')
    fig.canvas.draw()
    image_pil = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    image_drawn = torchvision_F.to_tensor(image_pil.convert(mode))
    plt.close()
    return image_drawn
"""

@torch.no_grad()
def draw_keypoints(opt, image, kp_pos, kp_depth, color=None):
    mode = "RGBA" if image.shape[0]==4 else "RGB"
    image_pil = torchvision_F.to_pil_image(image.cpu()).convert("RGBA")
    draw_pil = PIL.Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(draw_pil)
    kp_num = kp_pos.size(0)
    kp_alpha = torch.clamp_min(kp_depth, min=opt.vis.kp.min_opacity)
    cmap = matplotlib.cm.get_cmap(opt.vis.kp.cmap)
    for i, kp in enumerate(kp_pos):
        x1 = kp[0] - opt.vis.kp.size
        x2 = kp[0] + opt.vis.kp.size
        y1 = kp[1] - opt.vis.kp.size
        y2 = kp[1] + opt.vis.kp.size
        c = color or tuple(map(lambda x: round(255 * x), cmap(i / kp_num)[:-1]))
        c += (round(255 * kp_alpha[i].item()),)
        draw.ellipse([x1, y1, x2, y2], fill=c, outline=None, width=0)

    image_pil.alpha_composite(draw_pil)
    image_drawn = torchvision_F.to_tensor(image_pil.convert(mode))
    return image_drawn
