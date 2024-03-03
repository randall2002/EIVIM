import os


def generate_filename_with_identifier(base_filename, identifier, value):
#生成新的文件名。
    # Split the base filename to get name and extension
    name, extension = os.path.splitext(base_filename)
    # Format the value to ensure two decimal places for floats, replacing dot with underscore
    if isinstance(value, float):
        value_str = "{:0.2f}".format(value).replace('.', '_')
    else:
        value_str = str(value)
    # Construct the new filename with the identifier and its value
    new_filename = f"{name}_{identifier}_{value_str}{extension}"
    return new_filename

def main():

    # Example usage
    base_filename = 'U_net.pkl'
    identifier = 'alpha'
    value = 0.25

    # Generate the new filename
    new_filename = generate_filename_with_identifier(base_filename, identifier, value)
    print(new_filename)
    result_filename = generate_filename_with_identifier("result.csv", 'alpha', 0.1)
    print(result_filename)
    pass


if __name__ == '__main__':
    main()