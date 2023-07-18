import mitsuba as mi


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

# <float name="fov" value="25"/>

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
            <translate x="0" y="0" z="-0.20"/>
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

def render_single_point_cloud(out_path, kp, extr, intr, resolution):
    # fov = degrees(2 * np.arctan(resolution / (2 ** 0.5 * intr[0, 0])))
    fov = 53
    cam_loc = (-extr[..., :3, :3].transpose(-1, -2) @ extr[..., :3, 3:]).flatten()
    print(cam_loc)
    xml_segments = [XML_HEAD.format(*cam_loc.tolist(), fov, resolution, resolution)] # , fov, resolution, resolution)]# intr[0, 0], resolution, resolution)]
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