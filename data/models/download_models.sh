while getopts o: flag
do
    case "${flag}" in
        o) o=${OPTARG};;
    esac
done

gdrive_folder="https://drive.google.com/drive/u/0/folders/1Sn0GfpVJukFNLkc0cKZAEUPlV2DQlR9J";
gdown $gdrive_folder --output $o --folder 