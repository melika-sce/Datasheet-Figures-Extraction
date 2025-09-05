import os
import sys
from pdf_converter.converter import PDFToImageConverter

def main():
    """
    Main function to run the PDF to image conversion process for all PDFs
    in a specified directory.
    """
    # --- User Configuration ---
    # The script will look for PDFs in this directory relative to main.py
    INPUT_DIR = "input_pdfs"
    
    # The script will save the output image folders in this directory.
    OUTPUT_DIR = "pdf_images_output"
    # ------------------------

    # --- Script Logic ---
    # Create the input and output directories if they don't exist.
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all PDF files in the input directory.
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]

    # If no PDFs are found, print a helpful message and exit.
    if not pdf_files:
        print(f"No PDF files found in the '{os.path.abspath(INPUT_DIR)}' directory.")
        print("Please add one or more PDF files to this folder and run the script again.")
        sys.exit(0)

    print(f"Found {len(pdf_files)} PDF file(s) to process.")

    # Instantiate the converter.
    converter = PDFToImageConverter()
    
    successful_conversions = 0
    for pdf_filename in pdf_files:
        try:
            full_pdf_path = os.path.join(INPUT_DIR, pdf_filename)
            
            # Create a dedicated output subdirectory for this PDF.
            pdf_name_without_ext = os.path.splitext(pdf_filename)[0]
            pdf_output_dir = os.path.join(OUTPUT_DIR, pdf_name_without_ext)
            
            print(f"\n--- Processing: {pdf_filename} ---")
            
            # Perform the conversion for the current PDF.
            images = converter.convert(full_pdf_path, pdf_output_dir)
            
            print(f"Successfully converted {len(images)} pages.")
            print(f"Output saved in: {os.path.abspath(pdf_output_dir)}")
            successful_conversions += 1

        except (FileNotFoundError, RuntimeError) as e:
            print(f"An error occurred while processing {pdf_filename}: {e}", file=sys.stderr)
            continue
    
    print(f"\n--- Process Finished ---")
    print(f"Successfully processed {successful_conversions} out of {len(pdf_files)} PDF files.")

if __name__ == "__main__":
    main()