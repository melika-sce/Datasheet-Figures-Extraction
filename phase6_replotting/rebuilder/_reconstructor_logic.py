
import json
import math
import re
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import cv2

# --- 1. Helper Functions for Parsing ---
def parse_axis_tick_value(text_string):
    text_string = str(text_string).strip().lower()
    try: return float(text_string)
    except ValueError: pass
    multipliers = {'k': 1e3, 'm': 1e6, 'g': 1e9, 'µ': 1e-6, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12}
    if text_string.endswith('m') and not any(s in text_string for s in ['µm','nm','pm','em']):
        try: return float(text_string[:-1]) * 1e-3
        except ValueError: pass
    if len(text_string) > 1 and text_string[-1] in multipliers:
        try: return float(text_string[:-1]) * multipliers[text_string[-1]]
        except ValueError: pass
    elif len(text_string) > 2 and text_string[-2:] in multipliers: # For "us" if 's' is part of unit
        try: return float(text_string[:-2]) * multipliers[text_string[-2:]] # Check multipliers key carefully
        except (ValueError, KeyError): pass

    match_sci = re.match(r"([-+]?\d*\.?\d+)\s*(?:e|x10\^|\*10\^)\s*([-+]?\d+)", text_string)
    if match_sci:
        try: return float(match_sci.group(1)) * (10 ** int(match_sci.group(2)))
        except ValueError: pass
    text_string = text_string.replace('⁻', '-').replace('−', '-')
    match_pow = re.match(r"([-+]?\d*\.?\d+)\^([-+]?\d+)", text_string)
    if match_pow:
        try: return float(match_pow.group(1)) ** int(match_pow.group(2))
        except ValueError: pass
    return None

def parse_axis_label(label_text_parts):
    if not label_text_parts: return {"raw_text": "", "parsed_quantity": "", "parsed_symbol": "", "parsed_unit": "", "text_bbox_px": None}
    full_raw_text = " ".join([part['text'] for part in label_text_parts if part.get('text')])
    quantity, symbol, unit = "", "", ""
    text_for_symbol_quantity = full_raw_text
    unit_match = re.search(r"\(([^)]+?)\)(?!\s*\w)", full_raw_text)
    if unit_match:
        unit_candidate = unit_match.group(1).strip()
        if not (len(unit_candidate.split()) > 1 and any(c.islower() for c in unit_candidate)):
            unit = unit_candidate
            text_for_symbol_quantity = full_raw_text.replace(unit_match.group(0), "").strip()
    parts_by_comma = [p.strip() for p in text_for_symbol_quantity.split(',') if p.strip()]
    if len(parts_by_comma) > 1:
        potential_symbol = parts_by_comma[-1]
        if re.fullmatch(r"[A-Za-z]+[A-Za-z0-9_]*", potential_symbol) and len(potential_symbol) <= 5:
            symbol = potential_symbol; quantity = ",".join(parts_by_comma[:-1]).strip()
        else: quantity = text_for_symbol_quantity
    else: quantity = text_for_symbol_quantity
    if not symbol and re.fullmatch(r"[A-Za-z]+[A-Za-z0-9_]*", quantity) and len(quantity) <=5 :
        symbol = quantity; quantity = ""
    if symbol and symbol in quantity:
        quantity = re.sub(r'\b' + re.escape(symbol) + r'\b', '', quantity, flags=re.IGNORECASE).strip().rstrip(',').strip()
    return {"raw_text": full_raw_text, "parsed_quantity": quantity.strip(), "parsed_symbol": symbol.strip(), "parsed_unit": unit.strip(), "text_bbox_px": combine_bboxes([part['bbox'] for part in label_text_parts if part.get('bbox')])}

def parse_series_label(text_string):
    text_string = str(text_string).strip()
    match = re.match(r"([A-Za-z0-9_][A-Za-z0-9_\s\.\-]*[A-Za-z0-9_])\s*([=:]|\|?=)\s*(.+)", text_string)
    if match:
        param_part, value_part = match.group(1).strip(), match.group(3).strip()
        if not param_part.replace('.','',1).isdigit() and value_part:
            return {"raw_text": text_string, "parsed_parameter": param_part, "parsed_value_str": value_part}
    return {"raw_text": text_string, "parsed_parameter": None, "parsed_value_str": text_string}

def parse_legend_item(text_string):
    match = re.match(r"([A-Za-z0-9_]+)\s*=\s*(.*)", text_string)
    if match: return {"raw_text": text_string, "parsed_parameter": match.group(1).strip(), "parsed_value_string": match.group(2).strip()}
    return {"raw_text": text_string, "parsed_parameter": None, "parsed_value_string": text_string}

def combine_bboxes(bboxes):
    if not bboxes: return None
    valid_bboxes = [b for b in bboxes if b and len(b) == 4]
    if not valid_bboxes: return None
    return [min(b[0] for b in valid_bboxes), min(b[1] for b in valid_bboxes), max(b[2] for b in valid_bboxes), max(b[3] for b in valid_bboxes)]

def get_bbox_center(bbox):
    if not bbox or len(bbox) != 4: return (None, None)
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def is_likely_junk_ocr(text_string):
    if not text_string or not text_string.strip(): return True
    if re.fullmatch(r"[\|_\-.=\+/]{3,}", text_string) or re.fullmatch(r"([^\w\s])\1{2,}", text_string): return True
    if len(text_string) <= 2 and not re.search(r"[a-zA-Z0-9]", text_string): return True
    return False

def detect_axis_scale(sorted_ticks_with_parsed_values):
    if len(sorted_ticks_with_parsed_values) < 3: return "linear"
    values = [t['parsed_value'] for t in sorted_ticks_with_parsed_values if t['parsed_value'] is not None and t['parsed_value'] > 0]
    if len(values) < 3: return "linear"
    log_values = [math.log10(v) for v in values]
    log_diffs = [log_values[i+1] - log_values[i] for i in range(len(log_values)-1)]
    if not log_diffs: return "linear"
    mean_log_diff = sum(log_diffs) / len(log_diffs)
    if abs(mean_log_diff) < 1e-6 : return "linear" # effectively constant values, treat as linear
    variance = sum([(d - mean_log_diff)**2 for d in log_diffs]) / len(log_diffs)
    std_dev = math.sqrt(variance)
    if (std_dev / abs(mean_log_diff)) < 0.25: # Tolerance for variation in log steps
        # print(f"DEBUG_SCALE_DETECT: Detected LOG scale. Values: {values}")
        return "log"
    # print(f"DEBUG_SCALE_DETECT: Detected LINEAR scale. Values: {values}")
    return "linear"

# --- 2. Core Processing Functions ---
def process_axes(ocr_data, diagram_structure_data):
    axes_collection = []
    axis_id_counter = 1
    for axis_type in ["x_axis", "y_axis"]:
        axis_ocr_texts_all = [item for item in ocr_data.get('ocr_results', []) if item.get('associated_element') == axis_type]
        axis_structure_info = next((label for label in diagram_structure_data.get('labels', []) if label.get('class') == axis_type), None)
        if not axis_ocr_texts_all or not axis_structure_info: continue
        axis_obj = {"axis_id": f"{axis_type[0]}_{axis_id_counter}", "orientation": axis_type[0]}
        if axis_type[0] == 'x': axis_id_counter +=1
        title_parts, tick_texts_ocr = [], []
        for item in axis_ocr_texts_all:
            text_content = item.get('text', "")
            is_tick = False
            parsed_as_tick = parse_axis_tick_value(text_content)
            if parsed_as_tick is not None: is_tick = True
            elif not re.search(r"[a-zA-Z]{4,}", text_content): # Few letters
                if any(char.isdigit() for char in text_content) or len(text_content) < 6: is_tick = True
            if len(text_content.split()) > 4 and not is_tick and any(c.isalpha() for c in text_content): title_parts.append(item)
            elif is_tick: tick_texts_ocr.append(item)
            else: title_parts.append(item)
        sort_key_idx = 0 if axis_type == "x_axis" else 1
        title_parts.sort(key=lambda p: p['bbox'][sort_key_idx] if p.get('bbox') else 0)
        axis_obj['label_text'] = parse_axis_label(title_parts)
        ticks = []
        tick_texts_ocr.sort(key=lambda t: get_bbox_center(t.get('bbox'))[sort_key_idx] if t.get('bbox') else 0)
        processed_tick_texts = {}
        for tick_ocr in tick_texts_ocr:
            parsed_val = parse_axis_tick_value(tick_ocr['text'])
            if parsed_val is not None:
                center_coord = get_bbox_center(tick_ocr.get('bbox'))
                if center_coord[sort_key_idx] is None: continue
                dedup_key = (round(parsed_val, 4), round(center_coord[sort_key_idx] / 5.0))
                if dedup_key not in processed_tick_texts:
                    current_tick = {"raw_text": tick_ocr['text'], "parsed_value": parsed_val, "pixel_position": int(center_coord[sort_key_idx]), "text_bbox_px": tick_ocr.get('bbox')}
                    ticks.append(current_tick)
                    processed_tick_texts[dedup_key] = current_tick
        axis_obj['ticks'] = ticks
        axis_obj['scale_type'] = detect_axis_scale(axis_obj['ticks']) # Add scale type
        axis_obj['region_bbox_px'] = axis_structure_info.get('bbox') or axis_structure_info.get('bbox_normalized') # Prioritize absolute
        axes_collection.append(axis_obj)
    return axes_collection

def associate_lines_with_labels(line_pixel_data_list, plot_area_ocr_texts, plot_area_bbox):
    data_series_list, series_id_counter = [], 1
    valid_plot_area_ocr_texts = [item for item in plot_area_ocr_texts if item.get('text','').strip() and not is_likely_junk_ocr(item.get('text','').strip())]
    unassigned_line_indices = list(range(len(line_pixel_data_list)))
    for ocr_label_item in sorted(valid_plot_area_ocr_texts, key=lambda item: (item.get('bbox', [0,0,0,0])[1], item.get('bbox', [0,0,0,0])[0])):
        label_text, label_bbox = ocr_label_item.get('text'), ocr_label_item.get('bbox')
        if not label_text or not label_bbox: continue
        parsed_s_label = parse_series_label(label_text)
        label_center_x, label_center_y = get_bbox_center(label_bbox)
        if label_center_x is None: continue
        best_match_line_orig_idx, min_distance = -1, float('inf')
        for line_orig_idx in unassigned_line_indices:
            line_data_px = line_pixel_data_list[line_orig_idx]
            if not line_data_px or len(line_data_px) < 2: continue
            current_line_min_dist_to_label_center = float('inf')
            for i in range(len(line_data_px) - 1):
                p1,p2,label_pt_center=np.array(line_data_px[i]),np.array(line_data_px[i+1]),np.array([label_center_x,label_center_y])
                l2=np.sum((p1-p2)**2); dist=np.linalg.norm(label_pt_center - p1) if l2==0 else np.linalg.norm(label_pt_center - (p1 + max(0,min(1,np.dot(label_pt_center-p1,p2-p1)/l2))*(p2-p1)))
                current_line_min_dist_to_label_center = min(current_line_min_dist_to_label_center, dist)
            max_allowable_dist = float('inf')
            if plot_area_bbox and len(plot_area_bbox) == 4:
                plot_width,plot_height=plot_area_bbox[2]-plot_area_bbox[0],plot_area_bbox[3]-plot_area_bbox[1]
                if plot_width > 0 and plot_height > 0: max_allowable_dist = min(plot_width,plot_height)*0.30
                else: max_allowable_dist = 50 
            else: max_allowable_dist = 50 
            if current_line_min_dist_to_label_center < min_distance and current_line_min_dist_to_label_center < max_allowable_dist:
                min_distance, best_match_line_orig_idx = current_line_min_dist_to_label_center, line_orig_idx
        if best_match_line_orig_idx != -1:
            series_obj = {"series_id":f"series_{series_id_counter}","label_text":{"raw_text":parsed_s_label["raw_text"],"parsed_parameter":parsed_s_label.get("parsed_parameter"),"parsed_value_str":parsed_s_label.get("parsed_value_str"),"text_bbox_px":label_bbox},"line_pixel_coordinates":line_pixel_data_list[best_match_line_orig_idx],"calculated_data_points":[]}
            data_series_list.append(series_obj); series_id_counter += 1
            unassigned_line_indices.remove(best_match_line_orig_idx)
    for line_orig_idx in unassigned_line_indices:
        if line_pixel_data_list[line_orig_idx]:
            data_series_list.append({"series_id":f"series_{series_id_counter}","label_text":{"raw_text":"Unlabeled","text_bbox_px":None},"line_pixel_coordinates":line_pixel_data_list[line_orig_idx],"calculated_data_points":[]})
            series_id_counter +=1
    return data_series_list

def convert_pixels_to_data(pixel_coords, x_axis_obj, y_axis_obj):
    data_points = []
    if not x_axis_obj or not y_axis_obj or not x_axis_obj.get('ticks') or not y_axis_obj.get('ticks') or len(x_axis_obj['ticks'])<2 or len(y_axis_obj['ticks'])<2: return [[None,None] for _ in pixel_coords]
    x_ticks,y_ticks = sorted(x_axis_obj['ticks'],key=lambda t:t['pixel_position']),sorted(y_axis_obj['ticks'],key=lambda t:t['pixel_position'])
    is_x_log,is_y_log = x_axis_obj.get('scale_type')=='log',y_axis_obj.get('scale_type')=='log'
    for x_px,y_px in pixel_coords:
        x_data,y_data = None,None
        t1_x,t2_x = None,None
        for i in range(len(x_ticks)-1):
            if x_ticks[i]['pixel_position']<=x_px<=x_ticks[i+1]['pixel_position']:t1_x,t2_x=x_ticks[i],x_ticks[i+1]; break
        if not t1_x and x_ticks:
            if x_px<x_ticks[0]['pixel_position'] and len(x_ticks)>=2:t1_x,t2_x=x_ticks[0],x_ticks[1]
            elif x_px>x_ticks[-1]['pixel_position'] and len(x_ticks)>=2:t1_x,t2_x=x_ticks[-2],x_ticks[-1]
        if t1_x and t2_x:
            px1,px2,val1,val2=t1_x['pixel_position'],t2_x['pixel_position'],t1_x['parsed_value'],t2_x['parsed_value']
            if px1==px2:x_data=val1 if x_px==px1 else None
            else:
                ratio=(x_px-px1)/(px2-px1)
                if is_x_log and val1>0 and val2>0:log_val1,log_val2=math.log10(val1),math.log10(val2);x_data=10**(log_val1+ratio*(log_val2-log_val1))
                elif not is_x_log:x_data=val1+ratio*(val2-val1)
        t1_y,t2_y = None,None
        for i in range(len(y_ticks)-1):
            if y_ticks[i]['pixel_position']<=y_px<=y_ticks[i+1]['pixel_position']:t1_y,t2_y=y_ticks[i],y_ticks[i+1]; break
        if not t1_y and y_ticks:
            if y_px<y_ticks[0]['pixel_position'] and len(y_ticks)>=2:t1_y,t2_y=y_ticks[0],y_ticks[1]
            elif y_px>y_ticks[-1]['pixel_position'] and len(y_ticks)>=2:t1_y,t2_y=y_ticks[-2],y_ticks[-1]
        if t1_y and t2_y:
            py1,py2,val1_y,val2_y=t1_y['pixel_position'],t2_y['pixel_position'],t1_y['parsed_value'],t2_y['parsed_value']
            if py1==py2:y_data=val1_y if y_px==py1 else None
            else:
                ratio_y=(y_px-py1)/(py2-py1)
                if is_y_log and val1_y>0 and val2_y>0:log_val1_y,log_val2_y=math.log10(val1_y),math.log10(val2_y);y_data=10**(log_val1_y+ratio_y*(log_val2_y-log_val1_y))
                elif not is_y_log:y_data=val1_y+ratio_y*(val2_y-val1_y)
        data_points.append([x_data,y_data])
    return data_points

def process_plot_area(ocr_data, diagram_structure_data, axes_collection):
    plot_areas_list, plot_area_id_counter = [], 1
    plot_area_structure = next((label for label in diagram_structure_data.get('labels', []) if label.get('class') == 'plot_area'), None)
    if not plot_area_structure: return []
    plot_area_abs_pixel_bbox = plot_area_structure.get('bbox') 
    plot_area_origin_x, plot_area_origin_y = (plot_area_abs_pixel_bbox[0], plot_area_abs_pixel_bbox[1]) if plot_area_abs_pixel_bbox and len(plot_area_abs_pixel_bbox)==4 else (0,0)
    plot_area_display_bbox = plot_area_abs_pixel_bbox or plot_area_structure.get('bbox_normalized')
    if not plot_area_display_bbox or len(plot_area_display_bbox)!=4: return []
    plot_area_obj = {"plot_area_id":f"plot_area_{plot_area_id_counter}","region_bbox_px":plot_area_display_bbox,"associated_x_axis_id":next((ax['axis_id'] for ax in axes_collection if ax['orientation']=='x'),None),"associated_y_axis_id":next((ax['axis_id'] for ax in axes_collection if ax['orientation']=='y'),None),"data_series":[],"other_annotations_in_plot_area":[]}
    x_axis, y_axis = next((ax for ax in axes_collection if ax['axis_id']==plot_area_obj['associated_x_axis_id']),None),next((ax for ax in axes_collection if ax['axis_id']==plot_area_obj['associated_y_axis_id']),None)
    plot_area_ocr_texts,raw_line_pixel_data_list_relative = [item for item in ocr_data.get('ocr_results',[]) if item.get('associated_element')=='plot_area'],diagram_structure_data.get('lines',[])
    adjusted_line_pixel_data_list = []
    if plot_area_abs_pixel_bbox:
        for line_rel in raw_line_pixel_data_list_relative:
            adjusted_line_pixel_data_list.append([[p[0]+plot_area_origin_x,p[1]+plot_area_origin_y] for p in line_rel] if line_rel else [])
    else: adjusted_line_pixel_data_list=raw_line_pixel_data_list_relative
    data_series_with_pixels=associate_lines_with_labels(adjusted_line_pixel_data_list,plot_area_ocr_texts,plot_area_display_bbox)
    used_ocr_ids={id(s.get('label_text',{}).get('text_bbox_px')) for s in data_series_with_pixels if s.get('label_text',{}).get('text_bbox_px')}
    if x_axis and y_axis:
        for series in data_series_with_pixels:
            if series.get("line_pixel_coordinates"):series["calculated_data_points"]=convert_pixels_to_data(series["line_pixel_coordinates"],x_axis,y_axis)
    plot_area_obj["data_series"]=data_series_with_pixels
    for item in plot_area_ocr_texts:
        if item.get('bbox') and id(item['bbox']) not in used_ocr_ids:plot_area_obj["other_annotations_in_plot_area"].append({"raw_text":item.get('text'),"text_bbox_px":item['bbox']})
    plot_areas_list.append(plot_area_obj)
    return plot_areas_list

def process_legends(ocr_data, diagram_structure_data):
    legends_list, legend_id_counter = [], 1
    legend_box_structures = diagram_structure_data.get('legend_boxes', [])
    legend_ocr_texts_all = [item for item in ocr_data.get('ocr_results', []) if item.get('associated_element')=='legend_box' and item.get('text','').strip()]
    for leg_struct_idx, leg_struct in enumerate(legend_box_structures):
        region_bbox = leg_struct.get('bbox') or leg_struct.get('bbox_normalized')
        if not region_bbox or len(region_bbox)!=4: continue
        legend_obj = {"legend_id":f"legend_{legend_id_counter}","region_bbox_px":region_bbox,"title_text":{"raw_text":"","text_bbox_px":None},"items":[]}
        current_legend_texts_in_box=[t for t in legend_ocr_texts_all if t.get('bbox') and bbox_is_inside(t['bbox'],legend_obj['region_bbox_px'])]
        if not current_legend_texts_in_box: continue
        current_legend_texts_in_box.sort(key=lambda t:(t['bbox'][1],t['bbox'][0]))
        item_texts_for_grouping = current_legend_texts_in_box
        if current_legend_texts_in_box and ":" in current_legend_texts_in_box[0].get('text',''):
            legend_obj['title_text']={"raw_text":current_legend_texts_in_box[0]['text'],"text_bbox_px":current_legend_texts_in_box[0]['bbox']}
            item_texts_for_grouping=current_legend_texts_in_box[1:]
        if item_texts_for_grouping:
            i=0
            while i<len(item_texts_for_grouping):
                item1=item_texts_for_grouping[i];current_item_text_parts=[item1['text']];current_item_bboxes=[item1['bbox']]
                if i+1<len(item_texts_for_grouping):
                    item2=item_texts_for_grouping[i+1]
                    y_c1,y_c2=get_bbox_center(item1['bbox'])[1],get_bbox_center(item2['bbox'])[1]
                    x_e1,x_s2=item1['bbox'][2],item2['bbox'][0];h1=(item1['bbox'][3]-item1['bbox'][1])
                    if len(item1['text'].split())<=2 and not "=" in item1['text'] and item2['text'].strip().startswith("=") and y_c1 is not None and y_c2 is not None and h1>0 and abs(y_c1-y_c2)<h1*0.7 and (x_s2-x_e1)<30 and (x_s2-x_e1)>-10:
                        current_item_text_parts.append(item2['text'].strip());current_item_bboxes.append(item2['bbox']);i+=1
                full_item_text=" ".join(current_item_text_parts);full_item_text=re.sub(r'\s*=\s*','=',full_item_text).strip()
                parsed_item=parse_legend_item(full_item_text);parsed_item['text_bbox_px']=combine_bboxes(current_item_bboxes)
                legend_obj['items'].append(parsed_item);i+=1
        if legend_obj['title_text']['raw_text'] or legend_obj['items']:legends_list.append(legend_obj);legend_id_counter+=1
    return legends_list

def bbox_is_inside(inner_bbox, outer_bbox, overlap_threshold=0.1):
    if not inner_bbox or not outer_bbox or len(inner_bbox)!=4 or len(outer_bbox)!=4: return False
    try:inner_bbox,outer_bbox=[float(c) for c in inner_bbox],[float(c) for c in outer_bbox]
    except(ValueError,TypeError):return False
    ixmin,iymin,ixmax,iymax=max(inner_bbox[0],outer_bbox[0]),max(inner_bbox[1],outer_bbox[1]),min(inner_bbox[2],outer_bbox[2]),min(inner_bbox[3],outer_bbox[3])
    iw,ih=max(0,ixmax-ixmin),max(0,iymax-iymin);inter_area=iw*ih
    inner_w,inner_h=inner_bbox[2]-inner_bbox[0],inner_bbox[3]-inner_bbox[1]
    if inner_w<=0 or inner_h<=0: return False
    inner_area=inner_w*inner_h
    return(inter_area/inner_area)>=overlap_threshold


def reconstruct_digital_diagram(ocr_data, diagram_structure_data):
    """
    Main orchestration function to build the final digital diagram object.
    MODIFIED: This function now accepts dictionary objects directly instead of file paths.
    """
    # The original file loading block is removed.
    
    # All the logic below this point is identical to your original script.
    try:
        digital_diagram = {
            "diagram_metadata": {
                "source_pdf_name": diagram_structure_data.get("pdf_name", "N/A"),
                "source_page_number": diagram_structure_data.get("page_number", -1),
                "diagram_id_on_page": diagram_structure_data.get("diagram_id", -1),
                "original_image_width_px": diagram_structure_data.get("image_width", -1),
                "original_image_height_px": diagram_structure_data.get("image_height", -1),
                "detected_diagram_bbox_on_page_px": diagram_structure_data.get("diagram_bbox", [])
            },
            "axes_collection": [], 
            "plot_areas": [], 
            "legends": []
        }
        
        digital_diagram["axes_collection"] = process_axes(ocr_data, diagram_structure_data)
        digital_diagram["plot_areas"] = process_plot_area(ocr_data, diagram_structure_data, digital_diagram["axes_collection"])
        digital_diagram["legends"] = process_legends(ocr_data, diagram_structure_data)
        
        return digital_diagram

    except Exception as e:
        # Using print for direct feedback as in the original script
        print(f"Error during reconstruction process: {e}")
        return None


def save_digital_diagram(digital_diagram_data, output_path):
    try:
        with open(output_path, 'w', encoding='utf-8') as f: json.dump(digital_diagram_data, f, indent=2)
    except IOError as e: print(f"Error saving digital diagram to {output_path}: {e}")

def visualize_digital_diagram_on_ax(ax_target, digital_diagram_data, title="Digital Diagram"):
    ax_reconstructed = ax_target; ax_reconstructed.clear()
    any_series_plotted_with_label = False
    if digital_diagram_data.get("plot_areas"):
        for plot_area in digital_diagram_data["plot_areas"]:
            for series in plot_area.get("data_series", []):
                series_label_data = series.get("label_text", {}); raw_series_label = series_label_data.get("raw_text", "").strip()
                display_label = raw_series_label if raw_series_label and raw_series_label.lower() not in ["unlabeled", "unnamed series"] else None
                if series.get("calculated_data_points"):
                    points = [p for p in series["calculated_data_points"] if p[0] is not None and p[1] is not None]
                    if points:
                        x_vals,y_vals=[p[0] for p in points],[p[1] for p in points]
                        ax_reconstructed.plot(x_vals,y_vals,marker='.',markersize=4,linestyle='-',linewidth=2,label=display_label)
                        if display_label: any_series_plotted_with_label = True
    x_axis_info=next((ax for ax in digital_diagram_data.get("axes_collection",[]) if ax['orientation']=='x'),None)
    y_axis_info=next((ax for ax in digital_diagram_data.get("axes_collection",[]) if ax['orientation']=='y'),None)
    if x_axis_info:
        if x_axis_info.get('label_text',{}).get('raw_text'): ax_reconstructed.set_xlabel(x_axis_info['label_text']['raw_text'],fontsize=8)
        valid_x_ticks=[t for t in x_axis_info.get('ticks',[]) if t.get('parsed_value') is not None and t.get('raw_text') is not None]
        if valid_x_ticks:
            ax_reconstructed.set_xticks([t['parsed_value'] for t in valid_x_ticks]);ax_reconstructed.set_xticklabels([t['raw_text'] for t in valid_x_ticks],fontsize=7)
            if len(valid_x_ticks)>=1:ax_reconstructed.set_xlim(min([t['parsed_value'] for t in valid_x_ticks]),max([t['parsed_value'] for t in valid_x_ticks]))
        if x_axis_info.get('scale_type')=='log':ax_reconstructed.set_xscale('log')
    if y_axis_info:
        if y_axis_info.get('label_text',{}).get('raw_text'): ax_reconstructed.set_ylabel(y_axis_info['label_text']['raw_text'],fontsize=8)
        valid_y_ticks=[t for t in y_axis_info.get('ticks',[]) if t.get('parsed_value') is not None and t.get('raw_text') is not None]
        if valid_y_ticks:
            ax_reconstructed.set_yticks([t['parsed_value'] for t in valid_y_ticks]);ax_reconstructed.set_yticklabels([t['raw_text'] for t in valid_y_ticks],fontsize=7)
            if len(valid_y_ticks)>=1:ax_reconstructed.set_ylim(min([t['parsed_value'] for t in valid_y_ticks]),max([t['parsed_value'] for t in valid_y_ticks]))
        if y_axis_info.get('scale_type')=='log':ax_reconstructed.set_yscale('log')
    if any_series_plotted_with_label:ax_reconstructed.legend(fontsize='xx-small',loc='best')
    ax_reconstructed.set_title(title,fontsize=9);ax_reconstructed.grid(True,linestyle='--',alpha=0.7);ax_reconstructed.set_aspect('auto')

# --- Your Main Script's process_diagram_folder ---
def is_text_vertical_cv(ocr_result): # Renamed to avoid conflict if you also have the other is_text_vertical
    if ocr_result.get('associated_element', '') != 'y_axis': return False
    bbox = ocr_result.get('bbox')
    if bbox and len(bbox) == 4:
        width = abs(bbox[2] - bbox[0]); height = abs(bbox[3] - bbox[1])
        if width > 0 and height / width > 1.8:
            words = ocr_result.get('words', [])
            if not words and len(ocr_result.get('text','')) > 2 : return True
            if len(words) > 1 and sum(1 for w in words if len(w.get('text',''))==1)/len(words) > 0.5: return True
            elif len(words) == 1 and len(words[0].get('text','')) > 2 : return True 
    return False

def process_diagram_folder_main(folder_path): # Renamed to avoid conflict
    for filename in os.listdir(folder_path):
        if filename.startswith("diagram_") and (filename.endswith(".jpg") or filename.endswith(".png")):
            base_name_for_json, file_extension = os.path.splitext(filename)
            diagram_num = base_name_for_json.replace("diagram_", "")
            image_path = os.path.join(folder_path, filename)
            structure_json_path = os.path.join(folder_path, f"{base_name_for_json}.json")
            ocr_json_path = os.path.join(folder_path, f"ocr_{base_name_for_json}.json")

            if not (os.path.exists(structure_json_path) and os.path.exists(ocr_json_path)):
                print(f"JSONs for {filename} not found. Skipping...")
                continue

            digital_diagram_obj = reconstruct_digital_diagram(ocr_json_path, structure_json_path)
            if not digital_diagram_obj:
                print(f"Failed to reconstruct digital data for {filename}. Skipping plot.")
                continue
            
            try: 
                with open(structure_json_path, 'r') as f: diagram_data_for_annotator = json.load(f)
                with open(ocr_json_path, 'r') as f: ocr_data_for_annotator = json.load(f)
            except Exception as e:
                 print(f"Error re-loading JSONs for annotator for {filename}: {e}. Skipping...")
                 continue

            try:
                img_cv = cv2.imread(image_path)
                if img_cv is None: print(f"Error reading image {image_path}. Skipping..."); continue
                img_rgb_for_plot = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                
                fig = plt.figure(figsize=(28, 7)) 
                gs = gridspec.GridSpec(1, 4, width_ratios=[1.5, 1.5, 2.2, 0.5], wspace=0.15, hspace=0.2) 
                ax1, ax2, ax3, ax4 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]), fig.add_subplot(gs[3])

                ax1.imshow(img_rgb_for_plot); ax1.axis('off'); ax1.set_title('Original Diagram', fontsize=10)

                annotated_img_cv = img_cv.copy()
                img_height, img_width = annotated_img_cv.shape[:2]
                plot_area_bbox_for_annotator = next((l.get('bbox') for l in diagram_data_for_annotator.get('labels',[]) if l.get('class')=='plot_area'), None)
                plot_x1_offset, plot_y1_offset = (int(plot_area_bbox_for_annotator[0]), int(plot_area_bbox_for_annotator[1])) if plot_area_bbox_for_annotator and len(plot_area_bbox_for_annotator)==4 else (0,0)
                
                for det in diagram_data_for_annotator.get('labels', []) + diagram_data_for_annotator.get('legend_boxes', []):
                    if not det.get("bbox") or len(det["bbox"]) != 4: continue
                    x1,y1,x2,y2=[int(v) for v in det["bbox"]]; x1,y1,x2,y2=max(0,x1),max(0,y1),min(img_width,x2),min(img_height,y2)
                    if x2<=x1 or y2<=y1: continue
                    cls_name=det.get("class","?"); color=(255,0,0) if cls_name=="plot_area" else (0,0,255) if cls_name=="x_axis" else (0,255,0) if cls_name=="y_axis" else (255,255,0)
                    cv2.rectangle(annotated_img_cv,(x1,y1),(x2,y2),color,2); cv2.putText(annotated_img_cv,cls_name,(x1,y1-5 if y1>10 else y1+15),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
                
                for det in ocr_data_for_annotator.get('ocr_results', []): # OCR annotation for ax2
                    if not det.get("bbox") or len(det["bbox"]) != 4: continue
                    x1a,y1a,x2a,y2a=[int(v) for v in det["bbox"]]; x1a,y1a,x2a,y2a=max(0,x1a),max(0,y1a),min(img_width,x2a),min(img_height,y2a)
                    if x2a<=x1a or y2a<=y1a: continue
                    cv2.rectangle(annotated_img_cv,(x1a,y1a),(x2a,y2a),(180,180,180),1)
                    text_to_display_ocr=det.get("associated_element","none")[:10]
                    text_y_pos=y2a+12; text_y_pos=y1a-5 if text_y_pos > img_height-5 else text_y_pos
                    cv2.putText(annotated_img_cv,text_to_display_ocr,(x1a,text_y_pos),cv2.FONT_HERSHEY_SIMPLEX,0.3,(100,100,100),1)

                lines_for_annotator = diagram_data_for_annotator.get('lines', [])
                if lines_for_annotator:
                    for idx, line_pts in enumerate(lines_for_annotator):
                        if len(line_pts) < 2: continue
                        np.random.seed(idx); color = tuple(np.random.randint(50,220,3).tolist())
                        for i in range(len(line_pts)-1):
                            p1x,p1y=line_pts[i]; p2x,p2y=line_pts[i+1]
                            cv2.line(annotated_img_cv,(int(p1x+plot_x1_offset),int(p1y+plot_y1_offset)),(int(p2x+plot_x1_offset),int(p2y+plot_y1_offset)),color,2)
                ax2.imshow(cv2.cvtColor(annotated_img_cv, cv2.COLOR_BGR2RGB))
                ax2.axis('off'); ax2.set_title('Annotated Diagram (Detections)', fontsize=10)

                visualize_digital_diagram_on_ax(ax3, digital_diagram_obj, title="Digital Diagram")

                ax4.axis('off') 
                processed_legends_data = digital_diagram_obj.get("legends", [])
                actual_legend_box_text_lines, panel_4_title = [], ""
                if processed_legends_data:
                    temp_content_parts,has_embedded_title = [], False
                    for leg_data_item in processed_legends_data:
                        title,items=leg_data_item.get("title_text",{}).get("raw_text","").strip(),leg_data_item.get("items",[])
                        if title: temp_content_parts.append(f"{title}"); has_embedded_title = (":" in title and len(processed_legends_data)==1) or has_embedded_title
                        for item_detail in items:
                            item_text=item_detail.get('raw_text','').strip()
                            if item_text:temp_content_parts.append(f"  {item_text}")
                        if title or items: temp_content_parts.append("") 
                    actual_legend_box_text_lines=[line for line in temp_content_parts if line.strip()]
                    if actual_legend_box_text_lines and not has_embedded_title : panel_4_title = "Details" # Only add "Details" if no embedded title
                
                full_legend_text_for_panel4 = "\n".join(actual_legend_box_text_lines).strip()
                if full_legend_text_for_panel4:
                    if panel_4_title: ax4.set_title(panel_4_title,fontsize=8,loc='left',y=0.98,x=0.01) 
                    text_y_anchor = 0.95 if not panel_4_title else 0.90 # Lower text if panel has its own title
                    ax4.text(0.01,text_y_anchor,full_legend_text_for_panel4,transform=ax4.transAxes,fontsize=6.5,va='top',ha='left',linespacing=1.3,bbox=dict(boxstyle='round,pad=0.3',fc='ivory',ec='lightgrey',alpha=0.9))
                
                output_figure_path = os.path.join(folder_path, f"final_combined_vis_{diagram_num}{file_extension}")
                try: fig.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2) # Reduced padding further
                except ValueError: pass 
                plt.savefig(output_figure_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved combined visualization to: {output_figure_path}")
            except Exception as e:
                print(f"Error processing or plotting image {filename}: {e}")
                if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
    print("Processing complete.")

if __name__ == "__main__":
    # This now assumes all reconstructor functions are defined above in this same file.
    folder_path = "results/0 (32)/diagrams" # Your example folder
    
    if not os.path.exists(folder_path): # Create folder and dummy files for a quick test
        print(f"Input folder '{folder_path}' does not exist. Creating it with dummy files for testing.")
        os.makedirs(folder_path, exist_ok=True)
        try:
            # Create diagram_1.png
            img_d = Image.new('RGB',(1122,775),color='lightgray'); from PIL import ImageDraw; d=ImageDraw.Draw(img_d); d.text((50,50),"Dummy diagram_1.png for test",fill=(0,0,0)); img_d.save(os.path.join(folder_path,"diagram_1.png"))
            # Create diagram_1.json (structure) - using your provided structure
            diag1_structure = {"pdf_name": "Wolfspeed_C3M0016120K_data_sheet", "page_number": 4, "diagram_id": 1, "image_width": 1122, "image_height": 775, "diagram_bbox": [123, 382, 1245, 1157], "legend_boxes": [{"class": "legend_box", "bbox": [175, 51, 325, 161]}], "labels": [{"class": "plot_area", "bbox": [162, 42, 1067, 659]}, {"class": "y_axis", "bbox": [38, 42, 150, 661]}, {"class": "x_axis", "bbox": [133, 673, 1096, 757]}], "lines": [[[0,610],[100,500],[200,400],[300,350],[400,320],[500,300],[600,290],[700,285],[800,280],[900,270]],[[0,500],[100,400],[200,300],[300,250],[400,220],[500,200],[600,190],[700,185],[800,180],[900,170]],[[0,400],[100,300],[200,200],[300,150],[400,120],[500,100],[600,90],[700,85],[800,80],[900,70]],[[0,250],[100,200],[200,150],[300,100],[400,80],[500,60],[600,50],[700,45],[800,40],[900,35]],[[0,100],[100,80],[200,60],[300,40],[400,30],[500,25],[600,20],[700,18],[800,15],[900,12]]]} # Simplified lines
            with open(os.path.join(folder_path,"diagram_1.json"), 'w') as f: json.dump(diag1_structure, f)
            # Create ocr_diagram_1.json - using your provided OCR
            ocr1_data = {"pdf_name": "Wolfspeed_C3M0016120K_data_sheet", "page_number": 4, "diagram_id": 1, "image_width": 1122, "image_height": 775, "ocr_results": [{"text": "Drain-Source Current, IDs (A)","bbox": [39,185,70,535],"associated_element": "y_axis"},{"text": "250","bbox": [95,40,138,56],"associated_element": "y_axis"},{"text": "200","bbox": [95,162,138,177],"associated_element": "y_axis"},{"text": "150","bbox": [97,282,136,298],"associated_element": "y_axis"},{"text": "100","bbox": [97,402,136,420],"associated_element": "y_axis"},{"text": "50","bbox": [110,526,139,541],"associated_element": "y_axis"},{"text": "0","bbox": [124,646,140,662],"associated_element": "y_axis"},{"text": "Conditions:","bbox": [188,65,309,82],"associated_element": "legend_box"},{"text": "T\u2081","bbox": [187,97,205,116],"associated_element": "legend_box"},{"text": "= -40 \u00b0C","bbox": [212,96,292,112],"associated_element": "legend_box"},{"text": "tp = <200 \u03bcs","bbox": [188,126,323,145],"associated_element": "legend_box"},{"text": "VGS|= 15V","bbox": [433,59,521,83],"associated_element": "plot_area"},{"text": "VGS = 13V","bbox": [630,97,718,119],"associated_element": "plot_area"},{"text": "VGS = 11V","bbox": [819,107,906,128],"associated_element": "plot_area"},{"text": "VGS = 9V","bbox": [850,393,912,407],"associated_element": "plot_area"}, {"text": "VGS = 7V","bbox": [836,570,913,591],"associated_element": "plot_area"},{"text": "0.0","bbox": [147,682,183,697],"associated_element": "x_axis"},{"text": "2.0","bbox": [297,682,332,697],"associated_element": "x_axis"},{"text": "4.0","bbox": [447,682,484,697],"associated_element": "x_axis"},{"text": "6.0","bbox": [600,682,634,697],"associated_element": "x_axis"},{"text": "8.0","bbox": [750,681,786,697],"associated_element": "x_axis"},{"text": "10.0","bbox": [895,682,942,697],"associated_element": "x_axis"},{"text": "12.0","bbox": [1046,682,1095,697],"associated_element": "x_axis"},{"text": "Drain-Source Voltage, VDS (V)","bbox": [427,721,805,753],"associated_element": "x_axis"}]}
            with open(os.path.join(folder_path,"ocr_diagram_1.json"), 'w') as f: json.dump(ocr1_data, f)
            print("Created dummy image and JSON files based on your provided 'diagram_1' data.")
        except Exception as e_dum: print(f"Error creating dummy files: {e_dum}")

    process_diagram_folder_main(folder_path) # Call the renamed main processing loop