#### step 0: set data path    
`python config4cls.py`   

#### step 1: Generate 4 class label(car, big_vehicle, cyclist, pedestrian), Only need to do once   
`python convert_FineclsToCoarse.py --coarse_label_path ./coarse4_label_path/`   

#### step 2: roi filter for GT, Only need to do once   
`python roi_filter.py --txt_dir ./coarse4_label_path/label_2 --output_dir ./coarse4_label_path/label_2_filter`   

#### step 3: roi filter for pred   
input: $PRED_FILE (prediction folder name), output: $PRED_filter_file   

`PRED_FILE=$1`  
`PRED_FILETR_FILE=$PRED_FILE"_filter"`   
`python roi_filter.py --txt_dir ./result/$PRED_FILE --output_dir ./result/$PRED_FILETR_FILE`   

#### step 4: convert the ry of Pedestrian to GT value.   
input: $PRED_FILETR_FILE, output: RESULT_FILE=$PRED_FILETR_FILE"_ped"  
Due to the pedestrian category without annotation of ry, their ry prediction results were not considered.    

`python convert_Pedestrian_ry2GT_4cls.py --result_path ./result/$PRED_FILETR_FILE`    

#### step 5: compute the overall Score in V2X and save output txt into logfile   

`RESULT_FILE=$PRED_FILETR_FILE"_ped"`  
`LOG_FILE=$PRED_FILETR_FILE"_iou0.5.txt"`     
`python run_eval_distance_v2x_ground_4categories.py  --log_file $LOG_FILE --result_path ./result/$RESULT_FILE --iou_thresh 0.5'`   
