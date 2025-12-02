import os
import csv

def replace_string_in_csv(file_path, old_str, new_str):
    # read the .csv
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = [row for row in reader]

    # replace the str
    data = [[cell.replace(old_str, new_str) for cell in row] for row in data]

    # write the content into the original position
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def traverse_and_replace(root_folder, old_str, new_str):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(foldername, filename)
                replace_string_in_csv(file_path, old_str, new_str)

if __name__ == "__main__":
    folder_path = '../csv'

    old_string = '/share/share/Datasets/STAS/dataset/'
    new_string = '/'

    traverse_and_replace(folder_path, old_string, new_string)