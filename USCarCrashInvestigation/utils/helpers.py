from zipfile import ZipFile
import json

def read_config(config_file):
  """
  Reads a json config file and returns the loaded json
  :param config_file: path to the config file
  :return: loaded json file content
  """
  with open(f"{config_file}", "r") as f:
    config_content = json.load(f)
  return config_content

def deduplicate_df(dup_df):
  """
  De-duplicates a given dataframe
  :param dup_df: The DataFrame to be de-duplicated
  :return: De-duplicated dataframe
  """
  df = dup_df.dropDuplicates()
  return df

def load_csv(spark, file_path):
    """
    Read CSV data
    :param spark: spark instance
    :param file_path: path to the csv file
    :return: dataframe
    """
    return spark.read.option("header",True).csv(f"{file_path}")

def extract_zipped_content(zipped_file_path, unzipped_files_path):
  """
  Extracts all the files present within a zipped folder to another location
  :param zipped_file_path: Location to the Zipped file
  :param unzipped_files_path: Location to the Extracted files
  :return: None
  """
  # loading the temp.zip and creating a zip object
  with ZipFile(f"{zipped_file_path}", 'r') as zobject:

	# Extracting all the members of the zip
	# into a specific location.
    zobject.extractall(path=f"{unzipped_files_path}")
    print("Unzipping files successfully to : ",unzipped_files_path)

def write_output_with_format(df, file_path, file_format):
    """
    Write data frame to any given format
    :param df: dataframe to be written as output file
    :param file_path: output file path
    :param write_format: file format
    :return: None
    """

    df.write.format(file_format).mode('overwrite').option("header", "true").save(file_path)

def rdd_ops_list(out_df,column_name):
  """
  Converts a column into a string of comma separated values
  :param out_df:
  :param column_name:
  :return: string of comma separated values
  """
  list_of_str = ""
  for i in  out_df.select(f'{column_name}').rdd.flatMap(lambda x: x).collect():
        list_of_str += i + ", "
    
  return list_of_str.strip(", ")