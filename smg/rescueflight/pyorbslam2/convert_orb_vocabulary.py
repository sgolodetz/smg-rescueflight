from timeit import default_timer as timer

from smg.pyorbslam2 import ORBVocabulary


def main() -> None:
    start = timer()
    text_vocab: ORBVocabulary = ORBVocabulary()
    text_vocab.load_from_text_file("C:/orbslam2/Vocabulary/ORBvoc.txt")
    end = timer()
    print(f"Load from text: {end - start}s")

    start = timer()
    text_vocab.save_to_binary_file("C:/orbslam2/Vocabulary/ORBvoc.bin")
    end = timer()
    print(f"Save to binary: {end - start}s")

    start = timer()
    binary_vocab: ORBVocabulary = ORBVocabulary()
    binary_vocab.load_from_binary_file("C:/orbslam2/Vocabulary/ORBvoc.bin")
    end = timer()
    print(f"Load from binary: {end - start}s")


if __name__ == "__main__":
    main()
