# TFI-Cazcarra
Converting ER Diagrams to .sql scripts using neural networks.

## Pipeline

<div align="center">
    <img src="./data/images_extra/pipeline.png" width="800">
</div>


## Predictions
For predicting an image, the steps are the following:

1) Create an environment and install the packages in requirements.txt
2) Download the models. See models/ for more info.
3) Execute predict.py

Example usage:

```
python predict.py --img_path ./imagenes/imagen.png --path_to_save ./resultados/resultados_imagen.sql
```


## Inference parameters

Depending on the characteristics of each ERD or on the initial results we got from the predictions, we can adjust the values in the ```inference_params.yaml``` file to get better results.
Currently available parameters are:

### For 'tablas':
- **nms_threshold**: Non Maximum Suppression threshold for object 'tabla'. Unless there's some very specific case, it's better off to leave it with the default value (0.5)
- **score_threshold**: Filters every prediction with score confidence lesser than score_threshold variable. If the initial predictions are not good, one can set a good threshold by seeing the predictions with the ```--plot``` flag at the moment of predicting.
- **offset**: Offset when suppressing tables at the moment of finding connections between tables. Useful when the bounding box doesn't fully cover the tables.

### For 'cardinalidades':
- **nms_threshold**: Non Maximum Suppression threshold for object 'tabla'. Unless there's some very specific case, it's better off to leave it with the default value (0.5)
- **score_threshold**: Filters every prediction with score confidence lesser than score_threshold variable. If the initial predictions are not good, one can set a good threshold by seeing the predictions with the ```--plot``` flag at the moment of predicting.
- **distance_threshold**: Clean every object 'cardinalidad' if the nearest table is farther than ```distance_threshold``` pixels.

### For 'ocr':
- **lang**: Language of the ERD to transform. Conditions the matching of foreign keys and the transcription algorithm.
- **reescale_percent**: Percentage of reescale for tables. Default at 100 (no reescale).


## Example

Executing this ERD through the system

<div align="center">
    <img src="./data/images_extra/ejemplo_diagrama.png" width="800">
</div>
<br>

results into the following SQL code:

```
CREATE TABLE `tokens` (
  `token_id` INT(11) NOT NULL,
  `token` CHAR(64),
  `user_id` INT(11) NOT NULL,
  `token_expires` DATETIME,
  PRIMARY KEY (`token_id`)
);

CREATE TABLE `poems` (
  `poem_id` INT(11) NOT NULL,
  `title` VARCHAR(200),
  `poem` TEXT,
  `date_submitted` DATETIME,
  `category_id` INT(11) NOT NULL,
  `user_id` INT(11) NOT NULL,
  `date_approved` DATETIME,
  PRIMARY KEY (`poem_id`)
);

CREATE TABLE `categories` (
  `category_id` INT(11) NOT NULL,
  `category` VARCHAR(100),
  PRIMARY KEY (`category_id`)
);

CREATE TABLE `users` (
  `user_id` INT(11) NOT NULL,
  `first_name` VARCHAR(50),
  `last_name` VARCHAR(50),
  `email` VARCHAR(100),
  `username` VARCHAR(30),
  `pass_phrase` VARCHAR(500),
  `is_admin` TINYINT(4),
  `date_registered` DATETIME,
  `profile_pic` VARCHAR(30),
  `registration_confirmed` TINYINT(4),
  PRIMARY KEY (`user_id`)
);

ALTER TABLE `tokens` ADD FOREIGN KEY (`user_id`) REFERENCES `users`(`user_id`);
ALTER TABLE `poems` ADD FOREIGN KEY (`category_id`) REFERENCES `categories`(`category_id`);
ALTER TABLE `poems` ADD FOREIGN KEY (`user_id`) REFERENCES `users`(`user_id`);
```
