# Google Drive Setup
## 1. Create directory in root of google drive called 'output' (case sensitive).

## 2. Add COCO dataset share to your google drive account via this link: https://drive.google.com/drive/folders/1EVsLBRwT2njNWOrmBAhDHvvB8qrd9pXT?usp=sharing

## 3. Now COCO directory should be in your 'Shared with me' section of google drive

## 4. Create shortcut of COCO share to your drive, right click and select 'Add Shortcut to drive', choose root of 'My Drive'. Now you should see COCO directory under 'My Drive' section.

## 3. Mount Drive Code:
```python
from google colab import drive 
drive.mount('/content/gdrive') 
```
This will prompt you to go to googles OAUTH page and give you an oath key you must paste into the notebook and press enter.

