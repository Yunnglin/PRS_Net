SETLOCAL ENABLEDELAYEDEXPANSION
set /a count = 0
set /a size = 1500

for /d %%v in (../datasets/ShapeNetCore/skateboard/04225987/*) do (
    set /a count +=1
    md ..\data\!count!
    "D:\ThunderDownload\software\PCL\PCL 1.8.1\bin\pcl_mesh_sampling_release.exe" ..\datasets\ShapeNetCore\skateboard\04225987\%%v\model.obj ^
    ..\data\!count!\model.pcd -n_samples %size% -no_vis_result
    echo ------!count!---------
)

echo finish
ENDLOCAL