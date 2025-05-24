//#![cfg(ocvrs_has_module_imgproc)]
use std::error::Error;
use std::fs;
use std::time::Instant;

use clap::{Parser, Subcommand, arg};
use indicatif::{HumanDuration, ProgressBar};
use opencv::calib3d::get_optimal_new_camera_matrix;
use opencv::core::{
    Point2f, Point3f, Size, TermCriteria, TermCriteria_EPS, TermCriteria_MAX_ITER, Vector,
};
use opencv::imgcodecs::imwrite_def;
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc, not_opencv_branch_5, opencv_branch_5};
use serde::{Deserialize, Serialize};

opencv_branch_5! {
    use opencv::calib::{find_chessboard_corners_def, draw_chessboard_corners, calibrate_camera_def};
    use opencv::mod_3d::{undistort_def, init_undistort_rectify_map};
}

not_opencv_branch_5! {
    use opencv::calib3d::{find_chessboard_corners_def,  calibrate_camera_def, undistort_def};
}
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand, Debug)]
enum Action {
    Calibrate {
        #[arg(short, long)]
        calibration_dir: String,
        #[arg(short, long)]
        calibration_file: String,
    },
    Correct {
        #[arg(short, long)]
        calibration_file: String,
        #[arg(short, long)]
        correction_dir: String,
        #[arg(short, long)]
        output_dir: String,
    },
}

#[derive(Serialize, Deserialize)]
struct Calibration {
    camera_matrix: Vec<f64>,
    dist_coeffs: Vec<f64>,
}

// https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    match args.action {
        Action::Calibrate {
            calibration_dir,
            calibration_file,
        } => {
            // termination criteria
            let criteria = TermCriteria {
                typ: TermCriteria_EPS + TermCriteria_MAX_ITER,
                max_count: 30,
                epsilon: 0.001,
            };

            // prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            let width_dim = 11;
            let height_dim = 8;
            let objp_len = width_dim * height_dim;
            let objp = Vector::from_iter(
                (0..objp_len)
                    .map(|i| Point3f::new((i % width_dim) as f32, (i / height_dim) as f32, 0.)),
            );

            let mut objpoints = Vector::<Vector<Point3f>>::new(); // 3d point in real world space
            let mut imgpoints = Vector::<Vector<Point2f>>::new(); // 2d points in image plane.
            let pb = ProgressBar::new_spinner();
            pb.println("[1/3] process images");
            let started = Instant::now();
            fs::read_dir(&calibration_dir)?
                .flatten()
                .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "jpg"))
                .map(|entry| entry.path().to_string_lossy().to_string())
                .for_each(|image| {
                    // Arrays to store object points and image points from all the images.
                    pb.inc(1);
                    let img = imgcodecs::imread_def(&image).unwrap();
                    let mut gray = Mat::default();
                    imgproc::cvt_color_def(&img, &mut gray, imgproc::COLOR_BGR2GRAY).unwrap();

                    let mut corners = Vector::<Point2f>::default();
                    let ret = find_chessboard_corners_def(
                        &gray,
                        Size::new(width_dim, height_dim),
                        &mut corners,
                    )
                    .unwrap();
                    if ret {
                        imgproc::corner_sub_pix(
                            &gray,
                            &mut corners,
                            Size::new(11, 11),
                            Size::new(-1, -1),
                            criteria,
                        )
                        .unwrap();

                        // Draw and display corners
                        //draw_chessboard_corners(&mut img, Size::new(width_dim, height_dim), &corners, ret)?;
                        objpoints.push(objp.clone());
                        imgpoints.push(corners);
                        pb.set_message(format!(
                            "{image} processed. in progress for {}",
                            HumanDuration(started.elapsed())
                        ));
                    } else {
                        pb.println(format!("[!] chessboard not found for image {image}"));
                    }
                });

            pb.println("[2/3] compute calibration");
            let first_image = fs::read_dir(&calibration_dir)?
                .flatten()
                .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "jpg"))
                .map(|entry| entry.path().to_string_lossy().to_string())
                .next()
                .unwrap();

            let img = imgcodecs::imread_def(&first_image)?;
            let mut mtx = Mat::default();
            let mut dist = Mat::default();
            let mut rvecs = Vector::<Mat>::new();
            let mut tvecs = Vector::<Mat>::new();
            calibrate_camera_def(
                &objpoints,
                &imgpoints,
                img.size()?,
                &mut mtx,
                &mut dist,
                &mut rvecs, // rotation
                &mut tvecs, // translation
            )?;
            //use the calibration
            let width = img.cols();
            let height = img.rows();
            //println!("image dimensions : {} {}", width, height);
            let mtx = get_optimal_new_camera_matrix(
                &mtx,
                &dist,
                Size::new(width, height),
                1.0,
                Size::new(width, height),
                None,
                true,
            )?;

            // let mean_error = 0.0;
            // let mut imgpoints2 = Mat::default();
            // project_points_def(&objpoints, &rvecs, &tvecs, &mtx, &dist, &mut imgpoints2).unwrap();
            // mean_error +=
            //     norm(&imgpoints, &imgpoints2, NORM_L2).unwrap() / (imgpoints2.size() as f64);
            // println!("total error: {}", mean_error / objpoints.size());

            let calibration = Calibration {
                camera_matrix: mtx
                    .to_vec_2d()
                    .unwrap()
                    .iter()
                    .flat_map(|row| row.iter())
                    .cloned()
                    .collect::<Vec<f64>>(),
                dist_coeffs: dist
                    .to_vec_2d()
                    .unwrap()
                    .iter()
                    .flat_map(|row| row.iter())
                    .cloned()
                    .collect::<Vec<f64>>(),
            };
            pb.println(format!("[3/3] strore to file {calibration_file}"));
            fs::write(
                calibration_file,
                serde_json::to_string(&calibration).unwrap(),
            )
            .unwrap();
            pb.println(format!("done in {}", HumanDuration(started.elapsed())));
            pb.finish_and_clear();
        }
        Action::Correct {
            correction_dir,
            output_dir,
            calibration_file,
        } => {
            let calibraion: Calibration =
                serde_json::from_slice(&fs::read(calibration_file).unwrap()).unwrap();
            let mtx = Mat::new_rows_cols_with_data(3, 3, &calibraion.camera_matrix).unwrap();
            let dist = Mat::new_rows_cols_with_data(1, 5, &calibraion.dist_coeffs).unwrap();
            fs::read_dir(correction_dir)?
                .flatten()
                .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "jpg"))
                .for_each(|image| {
                    let img = imgcodecs::imread_def(&image.path().to_string_lossy()).unwrap();
                    let new_image = format!("u_{}", image.file_name().to_string_lossy());
                    println!("save new image {new_image}");

                    let mut dst_undistort = Mat::default();
                    undistort_def(&img, &mut dst_undistort, &mtx, &dist).unwrap();

                    imwrite_def(
                        format!("{}/{}", output_dir, new_image).as_str(),
                        &dst_undistort,
                    )
                    .unwrap();

                    // Using remapping
                    // let mut mapx = Mat::default();
                    // let mut mapy = Mat::default();
                    // init_undistort_rectify_map(
                    //     &mtx,
                    //     &dist,
                    //     &no_array(),
                    //     &no_array(),
                    //     img.size()?,
                    //     f32::opencv_type(),
                    //     &mut mapx,
                    //     &mut mapy,
                    // )?;
                    // let mut dst_remap = Mat::default();
                    // imgproc::remap_def(&img, &mut dst_remap, &mapx, &mapy, imgproc::INTER_LINEAR)?;
                    // imwrite_def(
                    //     format!("{}/u1_{}", output_dir, new_image).as_str(),
                    //     &dst_undistort,
                    // )?;
                });
        }
    }
    Ok(())
}
