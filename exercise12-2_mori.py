import os
import fitz


def extract_pdf(pdf_path, filename):
    pdf = fitz.open(pdf_path)

    for page_num in range(pdf.page_count):
        page = pdf.load_page(page_num)    
        image = page.get_pixmap()

        if os.path.exists(f'resource/{filename[:-4]}'):
            image.save(f"resource/{filename[:-4]}/{page_num + 1}.jpg")
        else:
            print("directory does't exist")


def create_dir(filename):

    # create each resource directory
    parent_dir = f'resource'
    resource_dir = f'{filename[:-4]}'
    resource_path = os.path.join(parent_dir, resource_dir)

    if not os.path.exists(resource_path):
        os.makedirs(resource_path)


def main():

    dir = 'pdfs_image_classification_task/pdfs'
    for filename in os.listdir(dir):
        pdf_path = os.path.join(dir, filename)

        create_dir(filename)
        
        extract_pdf(pdf_path, filename)
        print(f'{filename} page extracted successfully!')


if __name__=='__main__':
    main()