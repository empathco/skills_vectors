import os, psycopg2
from io import BytesIO
import logging
log = logging.getLogger(__name__)

# Extract files from the database
def extract_files_from_database():
    try:
        database_url = os.environ['DATABASE_URL']
        con = psycopg2.connect(database_url)
        cursor = con.cursor()

        log.debug('Connected')

        # Replace "your_table" and "file_path_column" with your actual table and column names
        query = "SELECT file_path FROM generic_jsa.files_storage"
        cursor.execute(query)

        # Fetch all rows
        rows = cursor.fetchall()

        # Specify the directory where you want to save the files
        output_directory = "output_directory"
        os.makedirs(output_directory, exist_ok=True)

        # Iterate through the rows and extract files
        for row in rows:
            file_path = row[0]  

            # Extract file name from the path
            file_name = os.path.basename(file_path)

            # Fetch the file content from the database
            cursor.execute(f"SELECT generic_jsa.files_storage.attach_data FROM generic_jsa.files_storage WHERE file_path = %s", (file_path,))
            file_content = cursor.fetchone()[0]  

            # Save the file to the output directory
            output_path = os.path.join(output_directory, file_name)
            with open(output_path, 'wb') as file:
                file.write(file_content)

        print("Files extracted successfully.")

    except (Exception, psycopg2.Error) as error:
        print("Error extracting files:", error)

    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        con.close()

# Main function
def main():
    extract_files_from_database()

if __name__ == "__main__":
    main()

