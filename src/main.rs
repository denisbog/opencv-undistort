//#![cfg(ocvrs_has_module_imgproc)]
use std::error::Error;
use std::fs;

use clap::{Parser, Subcommand, arg};
use opencv::calib3d::get_optimal_new_camera_matrix;
use opencv::core::{
    Point2f, Point3f, Size, TermCriteria, TermCriteria_EPS, TermCriteria_MAX_ITER, Vector, no_array,
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
    use opencv::calib3d::{find_chessboard_corners_def,  calibrate_camera_def, undistort_def, init_undistort_rectify_map};
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
struct Calibraion {
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

            let images: Vec<String> = fs::read_dir(calibration_dir)?
                .flatten()
                .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "jpg"))
                .map(|entry| entry.path().to_string_lossy().into())
                .collect();

            let mut objpoints = Vector::<Vector<Point3f>>::new(); // 3d point in real world space
            let mut imgpoints = Vector::<Vector<Point2f>>::new(); // 2d points in image plane.

            images.iter().for_each(|image| {
                println!("processing image {}", image);
                // Arrays to store object points and image points from all the images.

                let img = imgcodecs::imread_def(image).unwrap();
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
                    println!("processing image {} chessboard", image);
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
                } else {
                    println!("chessboard not found");
                }
            });

            let img = imgcodecs::imread_def(images.iter().next().unwrap())?;
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

            let calibration = Calibraion {
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
            fs::write(
                calibration_file,
                serde_json::to_string(&calibration).unwrap(),
            )
            .unwrap();
        }
        Action::Correct {
            correction_dir,
            output_dir,
            calibration_file,
        } => {
            let calibraion: Calibraion =
                serde_json::from_slice(&fs::read(calibration_file).unwrap()).unwrap();
            let mtx = Mat::new_rows_cols_with_data(3, 3, &calibraion.camera_matrix).unwrap();
            let dist = Mat::new_rows_cols_with_data(1, 5, &calibraion.dist_coeffs).unwrap();
            let images = fs::read_dir(correction_dir)?
                .flatten()
                .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "jpg"));
            for image in images {
                let first_image = image.path();
                let first_image = first_image;

                let new_image = format!(
                    "undistorted_{}",
                    first_image.file_name().unwrap().to_string_lossy()
                );
                let first_image = first_image.to_string_lossy();
                let img = imgcodecs::imread_def(&first_image)?;
                // Calibration

                //           println!("old {:?}, new {:?}", mtx, new_mtx);
                // Using cv.undistort()
                let mut dst_undistort = Mat::default();
                println!("save new image {new_image}");
                undistort_def(&img, &mut dst_undistort, &mtx, &dist)?;
                imwrite_def(
                    format!("{}/u_{}", output_dir, new_image).as_str(),
                    &dst_undistort,
                )?;

                // Using remapping
                let mut mapx = Mat::default();
                let mut mapy = Mat::default();
                init_undistort_rectify_map(
                    &mtx,
                    &dist,
                    &no_array(),
                    &no_array(),
                    img.size()?,
                    f32::opencv_type(),
                    &mut mapx,
                    &mut mapy,
                )?;
                let mut dst_remap = Mat::default();
                imgproc::remap_def(&img, &mut dst_remap, &mapx, &mapy, imgproc::INTER_LINEAR)?;
                imwrite_def(
                    format!("{}/u1_{}", output_dir, new_image).as_str(),
                    &dst_undistort,
                )?;
            }
        }
    }
    Ok(())
}
