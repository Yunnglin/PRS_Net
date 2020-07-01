SETLOCAL ENABLEDELAYEDEXPANSION
set /a count = 0
set /a size = 1600

for /d %%v in (../datasets/ShapeNetCore/bench/*) do (
    set /a count +=1
    md .\bench\!count!
    "D:\ThunderDownload\software\PCL\PCL 1.8.1\bin\pcl_mesh_sampling_release.exe" ^
    ..\datasets\ShapeNetCore\bench\%%v\model.obj ^
    .\bench\!count!\model.pcd -n_samples %size% -no_vis_result
    echo ------!count!---------
)

echo finish
ENDLOCAL