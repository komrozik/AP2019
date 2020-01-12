# import the TableHandler Class
import praktipy.tablehandler as th

# Parse in the Table
table = th.gen_from_txt("data.txt")

# Print the table
print(table)

# Generate a tex table
th.gen_tex_table(
        table, "example_table.tex",
        tex_caption="Put your laTeX caption here", 
        tex_label="Put your laTex label here", 
        subtables=0, #table splitten?
        precision=["2.3", 3, "1.9"], 
        
        midrule=2)#wie viele header zeilen 