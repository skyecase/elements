from copy import deepcopy
import re
import textwrap
from manim import *
from math import atan2, floor, ceil, pi
import json


def points_to_bezier_curve(points):
    obj = Circle()
    obj.set_points_smoothly(points)
    return obj


BASE_SHAPE_COLOR = GRAY_D
BASE_DOT_COLOR = GRAY_C
BASE_TEXT_COLOR = GRAY_C

HIGHLIGHT_COLOR = YELLOW_B
HIGHLIGHTED_TEXT_COLOR = GRAY_A

DEFAULT_FIGURE_BUFF = 0.35
WIDTH_TEXT_PCT = 0.4


class Bookmark:
    def __init__(self, tag_count, tag, text_offset):
        self.tag_count = tag_count
        self.tag = tag
        self.text_offset = text_offset

    def __repr__(self):
        return "<bookmark: %d, tag: %s, text_offset: %s>" % (
            self.tag_count,
            self.tag,
            self.text_offset,
        )

    def label_length(self):
        tokens = self.tag.split()
        type_ = tokens[0]

        if type_ == "circle:":
            return len(tokens[2])
        else:
            return len(tokens[1])


class Section:
    def __init__(
        self, text, duration, text_offset, appearing_shapes, disappearing_shapes
    ):
        self.duration = duration
        self.text = text
        self.text_offset = text_offset
        self.appearing_shapes = appearing_shapes
        self.disappearing_shapes = disappearing_shapes

    def __repr__(self):
        return (
            "<section: %s, duration: %.3f, text_offset: %d, appearing: %s, disappearing: %s>"
            % (
                self.text,
                self.duration,
                self.text_offset,
                str(self.appearing_shapes),
                str(self.disappearing_shapes),
            )
        )


def preprocess_tag(tag: str):
    tokens = tag.split()
    assert len(tokens) <= 3
    if tokens[1] in ["circle", "arc", "arcc", "gnomon"]:
        return " ".join([tokens[1], tokens[2], tokens[0]])
    elif tokens[1] in ["line", "polygon", "given", "point", "curve", "angle"]:
        return " ".join([tokens[1], tokens[0]])
    else:
        raise Exception("Unknown type in tag:", tag)


def reformat_prose(prose: str):
    def get_letters_from_tag(tag: str):
        tokens = tag.split(" ")
        type_ = tokens[0]
        if type_ in ["circle", "arc", "arcc"]:
            return tokens[2]
        else:
            return tokens[1]

    result = re.sub("[^\S\n\t]+\[Prop.*\]\.", ".", prose)
    result = re.sub("[^\S\n\t]+\[Prop.*\]\.", ",", prose)
    result = re.sub("\[Prop.*\]", "", result)
    result = re.sub("[^\S\n\t]+\[Post.*\]\.", ".", result)
    result = re.sub("[^\S\n\t]+\[Post.*\]\.", ",", result)
    result = re.sub("\[Post.*\]", "", result)
    result = re.sub("[^\S\n\t]+\[Def.*\]\.", ".", result)
    result = re.sub("[^\S\n\t]+\[Def.*\]\.", ",", result)
    result = re.sub("\[Def.*\]", "", result)
    result = re.sub("[^\S\n\t]+\[C.N.*\]\.", ".", result)
    result = re.sub("[^\S\n\t]+\[C.N.*\]\.", ",", result)
    result = re.sub("\[C.N.*\]", "", result)
    result = re.sub("—", "---", result)
    result = re.sub("[^\S]+---", "---", result)
    result = re.sub("---[^\S]+", "---", result)
    result = re.sub("[^\S]+-", "-", result)
    result = re.sub("-[^\S]+", "-", result)
    result = re.sub("---", ", ", result)

    result = (
        result.replace(")", "")
        .replace("(", "")
        .replace("]", "")
        .replace("[", "")
        .replace("{", "")
        .replace("}", "")
    )

    tag_count = 1
    bookmarks = []
    tag_replaced = ""
    offset = 0
    
    while offset < len(result):
        char = result[offset]
        if char == "{":
            tag = result[offset + 1 :].split("}")[0]
            tag = preprocess_tag(tag)
            letters = get_letters_from_tag(tag)
            bookmarks.append(Bookmark(tag_count, tag, len(tag_replaced)))
            tag_replaced += letters
            tag_count += 1
            offset += len(tag) + 1
        elif char == "}":
            offset += 1
        else:
            tag_replaced += char
            offset += 1

    return tag_replaced, bookmarks


def make_point(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    elif isinstance(x, list):
        pass
    else:
        raise Exception("Invalid type")

    if len(x) == 2:
        x = x + [0.]
    elif len(x) == 3:
        pass
    else:
        raise Exception("Invalid length")

    return np.array(x)


def transpose_label(coor, arr, size):
    mode = (arr[0] + 8) % 8

    if len(arr) == 1:
        l = 3
    else:
        l = arr[1] * 3

    def proj(mode, l):
        if mode == 0:
            transform_ = [0.1 * l, -0.1 * l]
        elif mode == 1:
            transform_ = [-0.5, -0.2 * l]
        elif mode == 2:
            transform_ = [-1 - 0.2 * l, -0.2 * l]
        elif mode == 3:
            transform_ = [-1 - 0.2 * l, 0.5]
        elif mode == 4:
            transform_ = [-1, 1]
        elif mode == 5:
            transform_ = [-0.5, 1 + 0.1 * l]
        elif mode == 6:
            transform_ = [0.3 * l, 0.9 + 0.1 * l]
        elif mode == 7:
            transform_ = [0.2 * l, 0.5]
        else:
            transform_ = [0.1 * l, -0.1 * l]
        transform_[0] += 0.5
        transform_[1] -= 0.3
        return make_point(transform_)

    if isinstance(mode, float):
        a = proj(ceil(mode), l)
        b = proj(floor(mode), l)
        t = mode - floor(mode)
        transform_ = b + (a - b) * t
    else:
        transform_ = proj(mode, l)

    transform_[0] *= size[0]
    transform_[1] *= -1 * size[1]

    return make_point(coor) + transform_


colors = [
    BLUE_B,
    GREEN_B,
    YELLOW_B,
    RED_B,
    MAROON_B,
]
current_color_count = 0


def get_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2

    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(dot_product)
    return angle


def convert_tag_to_shape_dict(tag, dict_):
    tokens = tag.split(" ")
    type_ = tokens[0]
    letters = tokens[1]

    if type_ == "point":
        points = [dict_["points"][letters[0]]]
    elif type_ == "line":
        points = [dict_["points"][i] for i in letters]
    elif type_ == "polygon":
        if "polygonl" in dict_:
            if letters in dict_["polygonl"]:
                letters = dict_["polygonl"][letters]

        if isinstance(letters, str):
            points = [[dict_["points"][i] for i in letters]]
        elif isinstance(letters, list):
            points = [[make_point(i) for i in letters]]
        else:
            raise Exception()
    elif type_ == "curve":
        points = [[dict_["points"][i] for i in letters]]
    elif type_ == "angle":
        points = [dict_["points"][i] for i in letters]
    elif type_ == "circle":
        points = [dict_["points"][i] for i in tokens[2]]
        center = dict_["points"][tokens[1]]
        diameter = 2 * np.linalg.norm(center - points[0])
        points = [center, diameter]
    elif type_ == "arc" or type_ == "arcc":
        points = [dict_["points"][i] for i in tokens[2]]
        to = points[0]
        from_ = points[1]
        center = dict_["points"][tokens[1]]
        points = [center, to, from_]
    else:
        raise Exception(type_)

    shape = [type_] + points

    return shape


def get_shape_animations(dict_, tag: str, point_labels):
    global current_color_count
    tokens = tag.split(" ")
    type_ = tokens[0]
    letters = tokens[1]
    current_color = colors[current_color_count % len(colors)]
    current_color_count += 1

    if type_ == "given":
        shapes = dict_["given"][letters]
        objs = []
        for shape in shapes:
            objs.append(
                create_shape(
                    shape,
                    stroke_width=4,
                    stroke_color=current_color,
                    fill_color=current_color,
                )
            )
        obj = VGroup(*objs)
    else:
        shape = convert_tag_to_shape_dict(tag, dict_)
        obj = create_shape(
            shape, stroke_width=4, stroke_color=current_color, fill_color=current_color
        )

    if type_ == "circle":
        letters = tokens[2]

    copy_letters = [
        point_labels[l].copy().set_fill(current_color)
        for l in letters
        if l in point_labels
    ]

    letters_highlight = AnimationGroup(*[FadeIn(i) for i in copy_letters], lag_ratio=1)
    letters_unhighlight = AnimationGroup(
        *[FadeOut(i) for i in copy_letters],
    )
    anim_in = AnimationGroup(Write(obj), letters_highlight)
    anim_out = AnimationGroup(FadeOut(obj), letters_unhighlight)

    return anim_in, anim_out, current_color


def preprocess_input_dict(dict_, figure_buff, scale=None, center=None):
    if scale is None and center is None:
        preprocess_input_dict(dict_, figure_buff, scale=1.0, center=np.array((0, 0, 0)))

        static_shapes = []
        for shape in dict_["shapes"]:
            obj = create_shape(shape)
            static_shapes.append(obj)
        all_shapes = VGroup(*static_shapes)

        xmin = all_shapes.get_center()[0] - all_shapes.width / 2
        xmax = all_shapes.get_center()[0] + all_shapes.width / 2
        ymin = all_shapes.get_center()[1] - all_shapes.height / 2
        ymax = all_shapes.get_center()[1] + all_shapes.height / 2

        xscale = (
            (1 - figure_buff)
            * (1 - WIDTH_TEXT_PCT)
            * config["frame_width"]
            / (xmax - xmin)
        )
        yscale = (1 - figure_buff) * config["frame_height"] / (ymax - ymin)
        coors_center = np.array(((xmax + xmin) / 2, (ymax + ymin) / 2, 0.0))
        coors_scale = min(xscale, yscale)
    else:
        coors_scale = scale
        coors_center = center

    def transform_coors(coor):
        if not isinstance(coor, np.ndarray):
            coor = np.array(coor)
        if len(coor) == 2:
            coor = np.append(coor, 0.0)

        result = coors_scale * (coor - coors_center)
        if scale == None and center == None:
            result[1] *= -1
            result += config["frame_width"] * WIDTH_TEXT_PCT / 2 * LEFT
        return result

    def transform_shape_coors(arr):
        type_ = arr[0]
        if type_ == "line":
            for idx in range(1, len(arr)):
                if not isinstance(arr[idx], (list, np.ndarray)):
                    continue
                arr[idx] = transform_coors(arr[idx])

        elif type_ == "polygon" or type_ == "curve":
            arr[1] = [transform_coors(i) for i in arr[1]]
        elif type_ == "circle":
            arr[1] = transform_coors(arr[1])
            arr[2] *= coors_scale
        elif type_ in ["arc", "arcc", "anglecurve", "gnomon"]:
            for idx in range(1, len(arr)):
                arr[idx] = transform_coors(arr[idx])
        else:
            raise Exception("Unkown shape type: " + type_)

    # Transform all coors
    for label, coor in dict_["points"].items():
        dict_["points"][label] = transform_coors(coor)

    for arr in dict_["shapes"]:
        transform_shape_coors(arr)

    if "given" in dict_:
        for _, arrs in dict_["given"].items():
            for arr in arrs:
                transform_shape_coors(arr)


def create_shape(shape, stroke_width=2, stroke_color=BASE_SHAPE_COLOR, fill_color=None):
    if fill_color == None:
        opacity = 0
    else:
        opacity = 0.75
    type_ = shape[0]
    if type_ == "line":
        points = [i for i in shape[1:] if isinstance(i, (list, np.ndarray))]
        dashed = False
        for i in shape:
            if isinstance(i, dict) and "dashed" in i and i["dashed"] == True:
                dashed = True

        objs = []
        for idx in range(len(points) - 1):
            line = Line(
                start=points[idx],
                end=points[idx + 1],
                stroke_width=stroke_width,
            ).set_color(stroke_color)
            if dashed:
                line = DashedVMobject(line)
            objs.append(line)
        obj = VGroup(*objs)
    elif type_ == "point":
        point = shape[1]
        obj = Dot(point).set_fill(fill_color)
    elif type_ == "polygon":
        points = shape[1]
        obj = (
            Polygon(*points, stroke_width=stroke_width)
            .set_color(stroke_color)
            .set_fill(fill_color, opacity=opacity)
        )
    elif type_ == "curve":
        points = shape[1]
        obj = (
            points_to_bezier_curve(points)
            .set_color(stroke_color)
            .set_stroke(width=stroke_width)
        )
    elif type_ == "circle":
        center = shape[1]
        radius = shape[2] / 2
        obj = (
            Circle(radius, stroke_width=stroke_width)
            .move_to(center)
            .set_color(stroke_color)
            .set_fill(fill_color, opacity=opacity)
        )
    elif type_ in ["arc", "arcc", "gnomon"]:
        center = shape[1]
        to, from_ = shape[2:]
        radius = np.linalg.norm(from_ - center)

        foc = from_ - center
        toc = to - center
        if type_ == "arc" or type_ == "gnomon":
            start_angle = atan2(toc[1], toc[0])
            angle = atan2(foc[1], foc[0]) - start_angle
            while angle > start_angle:
                angle -= 2 * pi
        else:
            start_angle = atan2(foc[1], foc[0])
            angle = atan2(toc[1], toc[0]) - start_angle
            while angle < start_angle:
                angle += 2 * pi

        obj = Arc(
            start_angle=start_angle,
            angle=angle,
            arc_center=center,
            radius=radius,
            stroke_width=stroke_width,
        ).set_color(stroke_color)
        if type_ == "gnomon":
            obj = DashedVMobject(obj)
    elif type_ == "angle" or type_ == "anglecurve":
        points = shape[1:]
        v1 = points[2] - points[1]
        v2 = points[0] - points[1]

        radius = min(np.linalg.norm(v1), np.linalg.norm(v2)) * 0.2
        line1 = Line(start=points[0], end=points[1]).set_color(stroke_color)
        line2 = Line(start=points[1], end=points[2]).set_color(stroke_color)

        intersectee = (
            Polygon(*points)
            .set_color(stroke_color)
            .set_fill(fill_color, opacity=opacity)
        )
        circle = Circle(radius=radius).move_to(points[1])

        angle_obj = (
            Intersection(circle, intersectee)
            .set_color(stroke_color)
            .set_fill(fill_color, opacity=opacity)
        )
        if type_ == "anglecurve":
            obj = angle_obj
        else:
            obj = VGroup(VGroup(line1, line2), angle_obj)
    elif type_ == "rightangle":
        points = shape[1:]
        v1 = points[2] - points[1]
        v2 = points[0] - points[1]

        radius = min(np.linalg.norm(v1), np.linalg.norm(v2)) * 0.2
        line1 = Line(start=points[0], end=points[1]).set_color(stroke_color)
        line2 = Line(start=points[1], end=points[2]).set_color(stroke_color)

        p1 = points[1]
        v1_ = radius / np.linalg.norm(v1) * v1
        v2_ = radius / np.linalg.norm(v2) * v2

        polygon_points = [p1, p1 + v1_, p1 + v1_ + v2_, p1 + v2_]
        angle_obj = (
            Polygon(*polygon_points)
            .set_color(stroke_color)
            .set_fill(fill_color, opacity=opacity)
        )
        obj = VGroup(VGroup(line1, line2), angle_obj)
    else:
        raise Exception("Unkown shape type: " + type_)

    return obj


def generate_scene(
    dict_,
    figure_buff=DEFAULT_FIGURE_BUFF,
    dot_radius=0.03,
    point_label_font_size=35,
    stroke_width=2,
    name=None,
    animation_duration=2.0,  # Default duration for each animation step
):
    class MyScene(Scene):
        def construct(self):
            preprocess_input_dict(dict_, figure_buff=figure_buff)

            # Create points
            self.points = {
                label: Dot(radius=dot_radius).set_fill(BASE_DOT_COLOR).move_to(coor)
                for label, coor in dict_["points"].items()
                if label in dict_["letters"]
            }

            # Create point labels
            self.point_labels = {}
            for label, _ in dict_["letters"].items():
                if "smallletters" in dict_ and label in dict_["smallletters"]:
                    font_size = point_label_font_size * 0.7
                else:
                    font_size = point_label_font_size

                self.point_labels[label] = Tex(
                    r"\textbf{\textsf{" + label + r"}}",
                    font_size=font_size,
                    color=BASE_TEXT_COLOR,
                )

            # Transpose labels
            for label, point_label in self.point_labels.items():
                self.point_labels[label] = point_label.move_to(
                    transpose_label(
                        self.points[label].get_center(),
                        dict_["letters"][label],
                        [point_label.width, point_label.height],
                    ),
                )

            self.static_shapes = []
            for shape in dict_["shapes"]:
                obj = create_shape(shape, stroke_width=stroke_width)
                self.static_shapes.append(obj)

            initial_shapes = []
            # Add shapes
            for obj in self.static_shapes:
                initial_shapes.append(obj)

            # Add points
            for _, point in self.points.items():
                initial_shapes.append(point)
            # Add point labels
            for _, point_label in self.point_labels.items():
                initial_shapes.append(point_label)
            initial_shapes = VGroup(*initial_shapes)

            title = Tex(r"\textsf{" + "Proposition " + dict_["id"] + r"}")
            self.play(Write(title))
            self.wait()
            self.play(title.animate.to_corner(UL).scale(0.75))
            self.play(Write(initial_shapes), run_time=3)

            prose: str = dict_["prose"]
            lines = prose.split("\n")
            lines = [i for i in lines if i != ""]

            self.wait(0.5)

            for line_idx, line in enumerate(lines):
                processed_text, bookmarks = reformat_prose(line)
                
                print(f"Line {line_idx + 1}: {line}")
                print(f">> Processed: {processed_text}")

                # Display the text
                disp_text = processed_text.replace("{", r"\textbf{").replace("}", "}")
                disp_text = r"\\".join(textwrap.wrap(disp_text, width=22))
                disp_text = r"\flushleft\textsf{" + disp_text + r"}"
                
                par = Tex(disp_text, font_size=45).set_fill(BASE_SHAPE_COLOR)
                
                max_height = config["frame_height"] * (1 - figure_buff)
                if par.height > max_height:
                    par.scale_to_fit_height(max_height)
                
                par = par.align_on_border(LEFT, buff=0).shift(
                    config["frame_width"] * (1 - WIDTH_TEXT_PCT) * RIGHT
                )
                
                self.add(par)
                
                # Animate shapes based on bookmarks
                prev_anim_out = None
                for bookmark in bookmarks:
                    tag = bookmark.tag
                    try:
                        anim_in, anim_out, current_color = get_shape_animations(
                            dict_, tag, self.point_labels
                        )
                        
                        animations = []
                        if prev_anim_out is not None:
                            animations.append(prev_anim_out)
                        animations.append(anim_in)
                        
                        self.play(*animations, run_time=animation_duration)
                        prev_anim_out = anim_out
                        
                    except Exception as e:
                        print(f"Error processing bookmark {bookmark}: {e}")
                        continue
                
                if prev_anim_out is not None:
                    self.play(prev_anim_out, run_time=animation_duration)
                
                # Wait before removing text
                self.wait(1)
                self.play(FadeOut(par))

            self.play(Unwrite(initial_shapes), Unwrite(title), run_time=3)
            self.wait(0.5)

    if name is not None:
        MyScene.__name__ = name
    return MyScene


if __name__ == "__main__":
    config["disable_caching"] = True
    config["quality"] = "low_quality"
    
    # Example usage:
    # scene1 = generate_scene(
    #     json.loads(open("book-03-proposition-02.json").read()),
    #     name="B03P02",
    #     animation_duration=1.5,  # Customize animation speed
    # )()
    # scene1.render()
