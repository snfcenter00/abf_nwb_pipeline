import os
import pandas as pd
from pynwb import NWBHDF5IO
from datetime import datetime
import re
import signal
import sys

def parse_date(date_obj):
    """Parse various date formats and return YYYYMMDD string"""
    try:
        if hasattr(date_obj, 'strftime'):
            # datetime object
            return date_obj.strftime('%Y%m%d')
        else:
            # Try parsing string
            date_str = str(date_obj).strip()
            # Try ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
            match = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_str)
            if match:
                return f"{match.group(1)}{match.group(2)}{match.group(3)}"
            # Try YYYYMMDD format
            match = re.search(r'(\d{8})', date_str)
            if match:
                return match.group(1)
    except:
        pass
    return 'N/A'

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")

# Set timeout for NWB file reads (120 seconds)
NWB_READ_TIMEOUT = 120

# Get all NWB files in the directory
nwb_data_list = []
data_dir = input("Enter the directory path to scan for NWB files: ").strip()
if not os.path.isdir(data_dir):
    print(f"Error: '{data_dir}' is not a valid directory.")
    sys.exit(1)
print(f"Scanning directory: {os.path.abspath(data_dir)}")
processed_subjects = set()

# TEST MODE: Set to 3 to test first 3 subjects, or 0 for all
TEST_MODE = 0

# Load existing Excel file to check which subjects are already processed
try:
    excel_file = os.path.join(data_dir, 'Index Ephys.xlsx')
    df = pd.read_excel(excel_file)
    existing_subjects = set(df['Mouse Name'].astype(str).values)
    print(f"Found {len(existing_subjects)} existing entries in Excel file")
except:
    existing_subjects = set()
    print("Creating new Index Ephys.xlsx file")

if TEST_MODE > 0:
    print(f"\n[TEST MODE] Processing only first {TEST_MODE} subjects\n")

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('_icephys.nwb'):
            filepath = os.path.join(root, file)
            try:
                # Extract subject ID and session info from filename
                # Format: sub-XXXXXX_ses-XXXXXX_icephys.nwb
                match = re.search(r'sub-(\d+)_ses-(\d+[a-z]*)', file)
                if match:
                    subject_id = match.group(1)
                    
                    # Check if we've hit the test limit
                    if TEST_MODE > 0 and len(processed_subjects) >= TEST_MODE:
                        continue
                    
                    # Skip if already processed in this run OR already in Excel
                    if subject_id in processed_subjects or subject_id in existing_subjects:
                        continue
                    
                    session_id = match.group(2)
                    
                    print(f"Reading metadata from: {file}")
                    print(f"  File path: {filepath}")
                    print(f"  File size: {os.path.getsize(filepath) / (1024**2):.2f} MB")
                    
                    try:
                        io = NWBHDF5IO(filepath, 'r', load_namespaces=True)
                        
                        # Read only the root attributes instead of loading full file
                        session_start = io.read_builder().get('session_start_time')
                        if session_start is not None:
                            date_str = parse_date(session_start.data)
                        else:
                            date_str = 'N/A'
                        
                        # Try to read metadata more efficiently
                        print(f"  Loading NWB object...")
                        nwb = io.read()
                        print(f"  [OK] NWB loaded successfully")
                        lab = nwb.lab if hasattr(nwb, 'lab') and nwb.lab else 'N/A'
                        institution = nwb.institution if hasattr(nwb, 'institution') and nwb.institution else 'N/A'
                        
                        # Extract age if available
                        subject = nwb.subject if hasattr(nwb, 'subject') else None
                        age = 'N/A'
                        sex = 'N/A'
                        if subject:
                            if hasattr(subject, 'age') and subject.age:
                                age = str(subject.age)
                            if hasattr(subject, 'sex') and subject.sex:
                                sex_val = str(subject.sex).strip().upper()
                                # Map sex values to M/F
                                if sex_val.startswith('M'):
                                    sex = 'M'
                                elif sex_val.startswith('F'):
                                    sex = 'F'
                                else:
                                    sex = sex_val
                        
                        # Store data for this subject (first NWB file only)
                        analysis_directory = os.path.abspath(root)
                        nwb_data_list.append({
                            'subject_id': subject_id,
                            'session_id': session_id,
                            'date': date_str,
                            'lab': lab,
                            'institution': institution,
                            'age': age,
                            'sex': sex,
                            'filepath': filepath,
                            'filename': file,
                            'analysis_dir': analysis_directory
                        })
                        
                        processed_subjects.add(subject_id)
                        print(f"[OK] Processed: sub-{subject_id} - {date_str} - Lab: {lab} - Age: {age} - Sex: {sex} - Dir: {analysis_directory}")
                        
                    finally:
                        io.close()
                    
            except TimeoutException as e:
                print(f"[TIMEOUT] Skipping {file} - {e}")
                print(f"   This file is taking too long to read (>120 seconds)")
            except Exception as e:
                print(f"[ERROR] Error processing {filepath}: {e}")
                import traceback
                traceback.print_exc()

print(f"\n\nTotal NWB files found: {len(nwb_data_list)}")

# Now update the Index Ephys.xlsx file
print("\n=== Updating Index Ephys.xlsx ===")

# Reload Excel file (in case it was modified)
try:
    df = pd.read_excel(excel_file)
except:
    df = pd.DataFrame()

# Group by subject_id (one row per experiment)
# Since we already have one entry per subject, just use it directly
new_rows = []
for data in nwb_data_list:
    # Convert date format from YYYYMMDD to int
    try:
        date_formatted = int(data['date']) if data['date'] != 'N/A' else 'N/A'
    except ValueError:
        print(f"Warning: Invalid date format '{data['date']}' for subject {data['subject_id']}, using 'N/A'")
        date_formatted = 'N/A'
    
    new_row = {
        'Mouse Name': f"Human_{data['subject_id']}",
        'Date of Experimnet': date_formatted,
        'LAB': data['lab'],
        'Project ': 'Developmental Human Study',
        'Species': 'Human',
        'Model': 'mouse',
        'Culture_Subtype': None,
        'Line': None,
        'Cage Number': None,
        'Age': data['age'],
        'ZT': None,
        'Sex': data['sex'],
        'Recording Type': 'Intrinsic',
        'ZT.1': None,
        'Project .1': 'Developmental Human Study',
        'Experiment Type': 'Patch Clamp',
        'Gene of Interest': None,
        'Genetic Background': None,
        'Line_Type': None,
        'Clone': None,
        'Mutation': None,
        'Drug/Rescue': None,
        'Culture_Medium': None,
        'AAV': None,
        'Cre': None,
        'Light Cycle': None,
        'Day': None,
        'MD Situtation': None,
        'Housing Location': None,
        'Remove/Keep': 'KEEP',
        'WHY?': None,
        'Analysis Directory': data['analysis_dir'],
    }
    new_rows.append(new_row)

# Create dataframe from new rows
new_df = pd.DataFrame(new_rows)

# Combine with existing data
df_combined = pd.concat([df, new_df], ignore_index=True)

# Save to both Index Ephys.xlsx and compiled data.xlsx
try:
    df_combined.to_excel(excel_file, index=False)
    print(f"Index Ephys.xlsx has been updated successfully!")
except PermissionError:
    print(f"[WARNING] Could not write to {excel_file} - file may be open in another application. Please close it and try again.")

compiled_file = os.path.join(data_dir, 'compiled data.xlsx')
try:
    df_combined.to_excel(compiled_file, index=False)
    print(f"Compiled data.xlsx has been created successfully!")
except PermissionError:
    print(f"[WARNING] Could not write to {compiled_file} - file may be open in another application. Please close it and try again.")

print(f"Updated rows in Excel: {len(df_combined)}")
print(f"New rows added: {len(new_rows)}")


# Display summary of added data
print("\n=== Sample of Added Data ===")
for row in new_rows[:10]:
    print(f"Subject: {row['Mouse Name']}, Date: {row['Date of Experimnet']}, Lab: {row['LAB']}, Age: {row['Age']}, Dir: {row['Analysis Directory']}")
    
print("\n=== Experiment Summary ===")
for i, data in enumerate(nwb_data_list[:10], 1):
    print(f"{i}. Sub-{data['subject_id']} | Date: {data['date']} | Lab: {data['lab']} | Age: {data['age']} | Dir: {data['analysis_dir']}")

