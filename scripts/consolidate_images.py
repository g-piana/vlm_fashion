from pathlib import Path
import shutil

# main_dir = Path("/mnt/e/vlm_images/shoes_women_batch04_src/")
# target_dir = Path("/mnt/e/vlm_images/shoes_women_batch04/")
main_dir = Path("/mnt/e/vlm_images/bags_women_batch04_src/")
target_dir = Path("/mnt/e/vlm_images/bags_women_batch04/")
target_dir.mkdir(parents=True, exist_ok=True)

dir_list = [x for x in main_dir.iterdir() if x.is_dir()]
for dir in dir_list:
    print(f"Folder {dir}")
    source_dir = main_dir / dir
    file_list = source_dir.glob("*.webp")
    for t_file in file_list:
        shutil.copy(t_file, target_dir)
        print(f"\t {t_file} -> {target_dir}")
    