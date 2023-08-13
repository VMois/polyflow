rule task1:
    input:
        "inputs/input1.txt",
        "inputs/input2.txt"
    output:
        "./output_task1.txt"
    shell:
        "cat input1.txt input2.txt > outputs/task1.txt | echo Done >> outputs/task1.txt"

rule task2:
    input:
        "outputs/task1.txt"
    output:
        "outputs/task2.txt"
    shell:
        "cat outputs/task1.txt >> outputs/task2.txt"
