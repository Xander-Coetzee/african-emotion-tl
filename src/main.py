from load_and_pre_process import load_and_preprocess_brighter

def main():
    # language can be changed - currently hausa
    # african languages with emotions consistently represented are - hausa (hau) -> end of list :/
    language = 'hau'
    processed = load_and_preprocess_brighter(language)

    print("First 10 rows of the train data set pre processed:")
    print(processed['train'].head(10))


if __name__ == "__main__":
    main()