use crate::encoding::normalize_encoding;

fn tempfile_path() -> std::path::PathBuf {
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("mt_enc_test_{id}_{:?}", std::thread::current().id()));
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn utf8_file_unchanged() {
    let tmp = tempfile_path();
    let p = tmp.join("test.srt");
    let text = "1\n00:00:01,000 --> 00:00:02,000\nPółka z książkami\n";
    std::fs::write(&p, text.as_bytes()).unwrap();
    normalize_encoding(&p).unwrap();
    let result = std::fs::read_to_string(&p).unwrap();
    assert_eq!(result, text);
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn cp1250_converted_to_utf8() {
    let tmp = tempfile_path();
    let p = tmp.join("test.srt");
    let text = "Półka z książkami";
    let (encoded, _, _) = encoding_rs::WINDOWS_1250.encode(text);
    std::fs::write(&p, &*encoded).unwrap();

    normalize_encoding(&p).unwrap();

    let result = std::fs::read_to_string(&p).unwrap();
    assert!(result.contains("Półka"), "result: {result}");
    assert!(result.contains("książkami"), "result: {result}");
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn iso_8859_2_converted_to_utf8() {
    let tmp = tempfile_path();
    let p = tmp.join("test.srt");
    let text = "Źródło świata";
    let (encoded, _, _) = encoding_rs::ISO_8859_2.encode(text);
    std::fs::write(&p, &*encoded).unwrap();

    normalize_encoding(&p).unwrap();

    let result = std::fs::read_to_string(&p).unwrap();
    assert!(result.contains("Źródło"), "result: {result}");
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn utf8_bom_left_intact() {
    let tmp = tempfile_path();
    let p = tmp.join("test.srt");
    let text = b"\xef\xbb\xbfTest line";
    std::fs::write(&p, text).unwrap();

    normalize_encoding(&p).unwrap();

    let raw = std::fs::read(&p).unwrap();
    assert!(raw.starts_with(b"\xef\xbb\xbf"), "BOM should be preserved");
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn ascii_file_unchanged() {
    let tmp = tempfile_path();
    let p = tmp.join("test.srt");
    let text = "Hello world";
    std::fs::write(&p, text.as_bytes()).unwrap();

    normalize_encoding(&p).unwrap();

    let result = std::fs::read_to_string(&p).unwrap();
    assert_eq!(result, text);
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn count_polish_chars() {
    use crate::encoding::count_polish;
    assert_eq!(count_polish("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"), 18);
    assert_eq!(count_polish("Hello world"), 0);
    assert_eq!(count_polish("Cześć"), 2); // ś + ć (matches Python _count_polish)
    assert_eq!(count_polish(""), 0);
}
