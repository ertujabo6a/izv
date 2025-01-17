# Data Analysis and Visualization in Python Project

The aim of the project is to obtain, process, and analyze data available on the internet.
This project includes three parts, that are focused on differents aspects of analysis and visualization data in python.
The complete specification can be founded in each folder in **zadani.pdf**, but it's in Czech.

## Project Outline

### Part 1 - 20 points (izv-part01)

The first part of the project will focus on verifying mastery of techniques for efficient data processing, visualization, and collection.

- Efficient numerical computations (part01.py: distance())
- Basic visualization (part01.py: generate_graph(), generate_sinus())
- Data downloading and processing (part01.py: download_data())

### Part 2 - 20 points (izv-part02)

The aim of this part of the project is to conduct a basic analysis of the Czech Police Accident Statistics dataset. 
Data for individual years are downloaded in an XLS file and are available at the following link: [data_23_24.zip](https://ehw.fit.vutbr.cz/izv/data_23_24.zip).
A description of the data can be found on the webpage: [izv](https://ehw.fit.vutbr.cz/izv) or in dataset_description.xlxs
Although these files have an XLS extension, they are actually HTML files. This format is dictated by the data from public sources.

- Different perspectives on data (analysis.py: load_data(), parse_data())
- Advanced visualization of results (analysis.py: plot_state(), plot_alcohol())
- Drawing conclusions, understanding the data (analysis.py: plot_type())

### Overall Project - 60 points (izv-part03)

- Mapping data and performing operations on this data (geo.py)
- Correlation and prediction (stat.ipynb)
- Automatic generation of report sections (doc.py)
- Compilation into an analytical report (doc.pdf)
