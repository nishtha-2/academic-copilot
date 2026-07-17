from code_assistant import (
    explain_code,
    debug_code,
    generate_program
)

while True:

    print("\n" + "=" * 50)
    print("Academic Copilot")
    print("=" * 50)
    print("1. Explain Code")
    print("2. Debug Code")
    print("3. Generate Program")
    print("4. Exit")

    choice = input("\nEnter choice: ").strip()

    if choice == "4":
        print("\nGoodbye!")
        break

    elif choice == "3":

        question = input("\nEnter programming question:\n")

        answer = generate_program(question)

        print("\nAnswer:")
        print("=" * 50)
        print(answer)

    elif choice in ["1", "2"]:

        print("\nPaste your code below.")
        print("Type END on a new line when finished.\n")

        lines = []

        while True:

            line = input()

            if line.strip().upper() == "END":
                break

            lines.append(line)

        code = "\n".join(lines)

        if choice == "1":
            answer = explain_code(code)
        else:
            answer = debug_code(code)

        print("\nAnswer:")
        print("=" * 50)
        print(answer)

    else:
        print("\nInvalid choice. Please enter 1, 2, 3 or 4.")