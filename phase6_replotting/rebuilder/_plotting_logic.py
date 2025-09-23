import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def visualize_digital_diagram_on_ax(ax_target, digital_diagram_data, image_dimensions, title="Digital Diagram"):
    """
    DEFINITIVE VERSION: Renders the plot with all features:
    - Accepts image dimensions directly to prevent errors.
    - RESTORED: Correctly plots all data lines.
    - Uses high-contrast colors.
    - Implements direct line labeling with collision avoidance.
    - Implements position-aware classification and rendering of titles vs. axis labels.
    """
    fig = ax_target.get_figure()
    ax_reconstructed = ax_target
    ax_reconstructed.clear()
    
    img_height, img_width = image_dimensions

    # --- PLOTTING AND LINE LABELING LOGIC (RESTORED and CORRECT) ---
    distinct_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    all_plotted_lines = []
    if digital_diagram_data.get("plot_areas"):
        for plot_area in digital_diagram_data["plot_areas"]:
            for i, series in enumerate(plot_area.get("data_series", [])):
                if series.get("calculated_data_points"):
                    points = [p for p in series["calculated_data_points"] if p[0] is not None and p[1] is not None]
                    if points:
                        x_vals = np.array([p[0] for p in points])
                        y_vals = np.array([p[1] for p in points])
                        all_plotted_lines.append({'x': x_vals, 'y': y_vals, 'series': series})

    for i, line_data in enumerate(all_plotted_lines):
        series_label_data = line_data['series'].get("label_text", {})
        raw_series_label = series_label_data.get("raw_text", "").strip()
        display_label = raw_series_label if raw_series_label and raw_series_label.lower() not in ["unlabeled", "unnamed series"] else None
        
        x_vals, y_vals = line_data['x'], line_data['y']
        color = distinct_colors[i % len(distinct_colors)]
        
        # This is the critical line plotting call that was missing
        ax_reconstructed.plot(x_vals, y_vals, marker='.', markersize=3, linestyle='-', linewidth=1.5, color=color)

        if display_label:
            candidate_indices = [len(x_vals) // 4, len(x_vals) // 2, (3 * len(x_vals)) // 4]
            best_anchor_point = (x_vals[candidate_indices[1]], y_vals[candidate_indices[1]])
            max_min_distance = -1
            for idx in candidate_indices:
                candidate_x, candidate_y = x_vals[idx], y_vals[idx]
                min_dist_to_other_lines = float('inf')
                for j, other_line in enumerate(all_plotted_lines):
                    if i == j: continue
                    other_x, other_y = other_line['x'], other_line['y']
                    distances = np.sqrt((other_x - candidate_x)**2 + (other_y - candidate_y)**2)
                    min_dist_to_other_lines = min(min_dist_to_other_lines, np.min(distances))
                if min_dist_to_other_lines > max_min_distance:
                    max_min_distance = min_dist_to_other_lines
                    best_anchor_point = (candidate_x, candidate_y)
            y_min, y_max = ax_reconstructed.get_ylim()
            offset = (y_max - y_min) * 0.03
            ax_reconstructed.text(
                best_anchor_point[0], best_anchor_point[1] + offset,
                display_label, 
                color=color, 
                fontsize='small', 
                weight='bold',
                verticalalignment='bottom',
                horizontalalignment='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1')
            )

    # --- AXIS SETUP & TITLE CLASSIFICATION ---
    x_axis_info = next((ax for ax in digital_diagram_data.get("axes_collection", []) if ax['orientation'] == 'x'), None)
    y_axis_info = next((ax for ax in digital_diagram_data.get("axes_collection", []) if ax['orientation'] == 'y'), None)
    
    plot_area_bbox = digital_diagram_data.get("diagram_metadata", {}).get("plot_area_bbox_px")

    ax_reconstructed.set_xlabel('')
    ax_reconstructed.set_ylabel('')
    ax_reconstructed.set_title('')

    if y_axis_info:
        valid_y_ticks = [t for t in y_axis_info.get('ticks', []) if t.get('parsed_value') is not None]
        if valid_y_ticks:
            ax_reconstructed.set_yticks([t['parsed_value'] for t in valid_y_ticks])
            ax_reconstructed.set_yticklabels([t['raw_text'] for t in valid_y_ticks], fontsize=7)
        if y_axis_info.get('scale_type') == 'log':
            ax_reconstructed.set_yscale('log')
        
        full_y_label_text = "\n".join(p['text'] for p in sorted(y_axis_info.get('label_text', {}).get('raw_text_parts', []), key=lambda i: i['bbox'][0]))
        ax_reconstructed.set_ylabel(full_y_label_text, fontsize=8)

    if x_axis_info:
        valid_x_ticks = [t for t in x_axis_info.get('ticks', []) if t.get('parsed_value') is not None]
        if valid_x_ticks:
            ax_reconstructed.set_xticks([t['parsed_value'] for t in valid_x_ticks])
            ax_reconstructed.set_xticklabels([t['raw_text'] for t in valid_x_ticks], fontsize=7)
        if x_axis_info.get('scale_type') == 'log':
            ax_reconstructed.set_xscale('log')
        
        title_parts = []
        xlabel_parts = []
        if plot_area_bbox:
            plot_area_top_y = plot_area_bbox[1]
            for part in x_axis_info.get('label_text', {}).get('raw_text_parts', []):
                if part.get('bbox') and part['bbox'][3] < plot_area_top_y:
                    title_parts.append(part)
                else:
                    xlabel_parts.append(part)
        else:
            xlabel_parts = x_axis_info.get('label_text', {}).get('raw_text_parts', [])

        full_title = "\n".join(p['text'] for p in sorted(title_parts, key=lambda i: i['bbox'][1]))
        full_xlabel = "\n".join(p['text'] for p in sorted(xlabel_parts, key=lambda i: i['bbox'][1]))
        
        ax_reconstructed.set_title(full_title if full_title else title, fontsize=9)
        ax_reconstructed.set_xlabel(full_xlabel, fontsize=8)

    ax_reconstructed.grid(True, linestyle='--', alpha=0.7)
    ax_reconstructed.set_aspect('auto')


def create_combined_visualization(image_path, diagram_data_for_annotator, ocr_data_for_annotator, digital_diagram_obj, output_folder_path):
    try:
        base_name, file_extension = os.path.splitext(os.path.basename(image_path))
        diagram_num = base_name.replace("diagram_", "")
        
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            print(f"Error reading image {image_path}. Skipping...")
            return

        image_dimensions = img_cv.shape[:2]
        
        img_rgb_for_plot = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        fig = plt.figure(figsize=(28, 7))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1.5, 1.5, 2.2, 0.5])
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax4 = fig.add_subplot(gs[3])

        ax1.imshow(img_rgb_for_plot)
        ax1.axis('off')
        ax1.set_title('Original Diagram', fontsize=10)

        annotated_img_cv = img_cv.copy()
        plot_area_bbox_for_annotator = next((l.get('bbox') for l in diagram_data_for_annotator.get('labels', []) if l.get('class') == 'plot_area'), None)
        plot_x1_offset, plot_y1_offset = (int(plot_area_bbox_for_annotator[0]), int(plot_area_bbox_for_annotator[1])) if plot_area_bbox_for_annotator else (0, 0)
        
        for det in diagram_data_for_annotator.get('labels', []) + diagram_data_for_annotator.get('legend_boxes', []):
            if not det.get("bbox"): continue
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det.get("class", "?")
            color = (255, 0, 0) if cls_name == "plot_area" else (0, 0, 255) if cls_name == "x_axis" else (0, 255, 0) if cls_name == "y_axis" else (255, 255, 0)
            cv2.rectangle(annotated_img_cv, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img_cv, cls_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        for det in ocr_data_for_annotator.get('ocr_results', []):
            if not det.get("bbox"): continue
            x1a, y1a, x2a, y2a = [int(v) for v in det["bbox"]]
            cv2.rectangle(annotated_img_cv, (x1a, y1a), (x2a, y2a), (180, 180, 180), 1)
        
        lines_for_annotator = diagram_data_for_annotator.get('lines', [])
        if lines_for_annotator:
            for idx, line_pts in enumerate(lines_for_annotator):
                if len(line_pts) < 2: continue
                np.random.seed(idx)
                color = tuple(np.random.randint(50, 220, 3).tolist())
                for i in range(len(line_pts) - 1):
                    p1x, p1y = line_pts[i]; p2x, p2y = line_pts[i + 1]
                    cv2.line(annotated_img_cv, (int(p1x + plot_x1_offset), int(p1y + plot_y1_offset)), (int(p2x + plot_x1_offset), int(p2y + plot_y1_offset)), color, 2)
        
        ax2.imshow(cv2.cvtColor(annotated_img_cv, cv2.COLOR_BGR2RGB))
        ax2.axis('off')
        ax2.set_title('Annotated Diagram (Detections)', fontsize=10)

        visualize_digital_diagram_on_ax(ax3, digital_diagram_obj, image_dimensions, title="Digital Diagram")

        ax4.axis('off')
        processed_legends_data = digital_diagram_obj.get("legends", [])
        if processed_legends_data:
            actual_legend_box_text_lines = []
            for leg_data_item in processed_legends_data:
                title = leg_data_item.get("title_text", {}).get("raw_text", "").strip()
                items = leg_data_item.get("items", [])
                if title: actual_legend_box_text_lines.append(title)
                for item_detail in items:
                    actual_legend_box_text_lines.append(f"  {item_detail.get('raw_text', '').strip()}")
            full_legend_text = "\n".join(actual_legend_box_text_lines)
            ax4.text(0.01, 0.95, full_legend_text, transform=ax4.transAxes, fontsize=6.5, va='top', ha='left')
        
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.08, top=0.92, wspace=0.15)
        
        output_figure_path = os.path.join(output_folder_path, f"final_combined_vis_{diagram_num}{file_extension.replace('.jpg', '.png')}")
        plt.savefig(output_figure_path, dpi=200)
        plt.close(fig)
        print(f"  Saved combined visualization to: {output_figure_path}")

    except Exception as e:
        print(f"Error processing or plotting image {os.path.basename(image_path)}: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)