import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ._reconstructor_logic import visualize_digital_diagram_on_ax # Import from the sibling module


def is_text_vertical(ocr_result): # Your existing function
    if ocr_result.get('associated_element', '') != 'y_axis': return False
    bbox = ocr_result.get('bbox')
    if bbox and len(bbox) == 4:
        width = abs(bbox[2] - bbox[0]); height = abs(bbox[3] - bbox[1])
        if width > 0 and height / width > 1.8: # Adjusted ratio slightly for more inclusivity
            words = ocr_result.get('words', [])
            if not words and len(ocr_result.get('text','')) > 2: # Single long word, likely vertical
                return True
            if len(words) > 1 and sum(1 for w in words if len(w.get('text',''))==1)/len(words) > 0.5: return True # More than half are single char
            elif len(words) == 1 and len(words[0].get('text','')) > 2 : return True 
    return False


# --- IN _plotting_logic.py, REPLACE THE MAIN FUNCTION ---
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ._reconstructor_logic import visualize_digital_diagram_on_ax # Import from the sibling module

def create_combined_visualization(image_path, diagram_data_for_annotator, ocr_data_for_annotator, digital_diagram_obj, output_folder_path):
    """
    Creates and saves the 4-panel visualization for a single diagram.
    MODIFIED: This is an adaptation of the original 'process_diagram_folder' function.
    It no longer iterates through a folder but processes one diagram whose data is passed in directly.
    """
    # The original file-finding and data-loading loops are removed.
    
    # All the logic below this point is identical to your original script's main loop.
    try:
        base_name, file_extension = os.path.splitext(os.path.basename(image_path))
        diagram_num = base_name.replace("diagram_", "")

        img_cv = cv2.imread(image_path)
        if img_cv is None: 
            print(f"Error reading image {image_path}. Skipping...")
            return
        img_rgb_for_plot = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # --- Setup 4-panel plot ---
        fig = plt.figure(figsize=(28, 7)) 
        gs = gridspec.GridSpec(1, 4, width_ratios=[1.5, 1.5, 2.2, 0.5], wspace=0.15, hspace=0.2) 

        ax1 = fig.add_subplot(gs[0]) 
        ax2 = fig.add_subplot(gs[1]) 
        ax3 = fig.add_subplot(gs[2]) 
        ax4 = fig.add_subplot(gs[3]) 

        ax1.imshow(img_rgb_for_plot); ax1.axis('off'); ax1.set_title('Original Diagram', fontsize=10)

        # --- Annotation Logic (identical to original) ---
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

        for det in ocr_data_for_annotator.get('ocr_results', []):
            if not det.get("bbox") or len(det["bbox"]) != 4: continue
            x1a,y1a,x2a,y2a=[int(v) for v in det["bbox"]]
            x1a,y1a,x2a,y2a=max(0,x1a),max(0,y1a),min(img_width,x2a),min(img_height,y2a)
            if x2a<=x1a or y2a<=y1a: continue
            cv2.rectangle(annotated_img_cv, (x1a,y1a),(x2a,y2a),(180,180,180),1)
            text_to_display_ocr = det.get("associated_element", "none")[:10]
            text_y_pos = y2a + 12 
            if text_y_pos > img_height - 5: text_y_pos = y1a - 5
            cv2.putText(annotated_img_cv, text_to_display_ocr, (x1a, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,100), 1)

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

        # --- Digital Diagram Visualization (identical to original) ---
        visualize_digital_diagram_on_ax(ax3, digital_diagram_obj, title="Digital Diagram")

        # --- Legend/Conditions Panel (identical to original) ---
        ax4.axis('off') 
        processed_legends_data = digital_diagram_obj.get("legends", [])
        actual_legend_box_text_lines = []
        panel_4_title = ""
        if processed_legends_data:
            temp_content_parts = []
            for leg_data_item in processed_legends_data:
                title = leg_data_item.get("title_text", {}).get("raw_text", "").strip()
                items = leg_data_item.get("items", [])
                if title: temp_content_parts.append(f"{title}")
                if not panel_4_title and ":" not in title: panel_4_title = "Details"
                for item_detail in items:
                    item_text = item_detail.get('raw_text', '').strip()
                    if item_text: temp_content_parts.append(f"  {item_text}")
                if title or items: temp_content_parts.append("") 
            actual_legend_box_text_lines = [line for line in temp_content_parts if line.strip()]
        
        full_legend_text_for_panel4 = "\n".join(actual_legend_box_text_lines).strip()
        if full_legend_text_for_panel4:
            if panel_4_title: ax4.set_title(panel_4_title, fontsize=8, loc='left', y=0.98, x=0.01) 
            text_y_pos = 0.95 if not panel_4_title else 0.88
            ax4.text(0.01, text_y_pos, full_legend_text_for_panel4, transform=ax4.transAxes, fontsize=6.5, va='top', ha='left', linespacing=1.3,
                     bbox=dict(boxstyle='round,pad=0.3', fc='ivory', ec='lightgrey', alpha=0.9))
        
        # --- Save the final figure ---
        # MODIFIED: Use the output_folder_path argument
        output_figure_path = os.path.join(output_folder_path, f"final_combined_vis_{diagram_num}{file_extension}")
        try: fig.tight_layout(pad=0.3, h_pad=0.2, w_pad=0.2)
        except ValueError: pass 
        plt.savefig(output_figure_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved combined visualization to: {output_figure_path}")

    except Exception as e:
        print(f"Error processing or plotting image {os.path.basename(image_path)}: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)

