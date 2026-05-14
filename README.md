python image_alignment.py -d "D:/My New Data/Folder"

Suitable for larger displacements:
python image_alignment.py -d "D:/My New Data/Folder" --use_moments


You can modify the regular expression to satisfy different naming rules.

```python
pattern = re.compile(r"ROI (\d+) (Original|RIE)\.tiff")
```

