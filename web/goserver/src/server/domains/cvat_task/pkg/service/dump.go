package service

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	fp "path/filepath"
	"strconv"
	"strings"

	buildFindOne "server/db/pkg/handler/build/find_one"
	cvatTaskUpdateOne "server/db/pkg/handler/cvat_task/update_one"
	buildStatus "server/db/pkg/types/build/status"
	kitendpoint "server/kit/endpoint"

	"go.mongodb.org/mongo-driver/bson/primitive"

	assetFind "server/db/pkg/handler/asset/find"
	assetFindOne "server/db/pkg/handler/asset/find_one"
	assetUpdateUpsert "server/db/pkg/handler/asset/update_upsert"
	cvatTaskFind "server/db/pkg/handler/cvat_task/find"
	cvatTaskFindOne "server/db/pkg/handler/cvat_task/find_one"
	problemFindOne "server/db/pkg/handler/problem/find_one"
	t "server/db/pkg/types"
	typeAsset "server/db/pkg/types/type/asset"
	cvatApi "server/domains/cvat_task/pkg/third_part_api/cvat"
)

type DumpRequestData struct {
	Id primitive.ObjectID `bson:"_id" json:"id"`
}

const fileLogPrefix = "domains.cvat_task.pkg.service.dump."

func (s *basicCvatTaskService) Dump(ctx context.Context, req DumpRequestData) chan kitendpoint.Response {
	returnChan := make(chan kitendpoint.Response)
	go func() {
		defer close(returnChan)

		cvatTask := s.getCvatTask(req.Id)
		asset := s.getAsset(cvatTask.AssetId)
		if isFolder(asset) {
			childrenCvatTasks := s.getChildrenCvatTasks(asset, cvatTask.ProblemId)
			for _, childCvatTask := range childrenCvatTasks {
				s.Dump(ctx, DumpRequestData{Id: childCvatTask.Id})
			}
			return
		}

		buildFindOneResp := <-buildFindOne.Send(
			ctx,
			s.Conn,
			buildFindOne.RequestData{
				ProblemId: cvatTask.ProblemId,
				Status:    buildStatus.Tmp,
			},
		)
		tmpBuild := buildFindOneResp.Data.(buildFindOne.ResponseData)
		buildSplit := findBuildAssetSplit(tmpBuild, asset)
		if cvatTask.Status == "pullInProgress" {
			returnChan <- kitendpoint.Response{IsLast: true, Data: getCvatTaskAndAsset(cvatTask, asset, buildSplit), Err: nil}
			return
		}
		cvatTask.Status = "pullInProgress"
		resp := <-cvatTaskUpdateOne.Send(context.TODO(), s.Conn, cvatTask)
		cvatTask = resp.Data.(cvatTaskUpdateOne.ResponseData)
		returnChan <- kitendpoint.Response{IsLast: false, Data: getCvatTaskAndAsset(cvatTask, asset, buildSplit), Err: nil}

		problem := s.getProblem(cvatTask.ProblemId)
		tmpDirPath := mkTmpDir()
		defer removeTmpDir(tmpDirPath)

		dataZipPath := fmt.Sprintf("%s/data.zip", tmpDirPath)
		tmpBuildPath := makeTmpBuildPath(problem.Dir)
		annotationFileName := strconv.Itoa(cvatTask.Annotation.Id)
		unzipAnnotationPath := fmt.Sprintf("%s/%s.json", tmpBuildPath, annotationFileName)
		if !isCvatDatasetDumped(asset) {
			log.Println("Cvat dataset dump")
			cvatDataPath := mkUnzipDatasetDir(asset)
			_ = s.saveCvatDataPathToAsset(asset, cvatDataPath)
			if _, err := cvatApi.PrepareDataset(cvatTask.Annotation); err != nil {
				returnChan <- kitendpoint.Response{IsLast: true, Data: nil, Err: err}
				return
			}
			if err := cvatApi.CheckStatusPrepareDataset(cvatTask.Annotation); err != nil {
				returnChan <- kitendpoint.Response{IsLast: true, Data: nil, Err: err}
				return
			}
			if err := cvatApi.DownloadDatasetZip(cvatTask.Annotation, dataZipPath); err != nil {
				returnChan <- kitendpoint.Response{IsLast: true, Data: nil, Err: err}
				return
			}
			if err := UnzipDataset(dataZipPath, cvatDataPath); err != nil {
				returnChan <- kitendpoint.Response{IsLast: true, Data: nil, Err: err}
				return
			}
			err := fp.Walk(cvatDataPath,
				func(path string, info os.FileInfo, err error) error {
					if err != nil {
						return err
					}
					fmt.Println(path, info.Size())
					return nil
				})
			if err != nil {
				log.Println(err)
			}
		} else {
			log.Println("Cvat annotation export")
			if _, err := cvatApi.PrepareAnnotation(cvatTask.Annotation); err != nil {
				returnChan <- kitendpoint.Response{IsLast: true, Data: nil, Err: err}
				return
			}
			if err := cvatApi.CheckStatusPrepareAnnotation(cvatTask.Annotation); err != nil {
				returnChan <- kitendpoint.Response{IsLast: true, Data: nil, Err: err}
				return
			}
			if err := cvatApi.DownloadAnnotationZip(cvatTask.Annotation, dataZipPath); err != nil {
				returnChan <- kitendpoint.Response{IsLast: true, Data: nil, Err: err}
				return
			}
		}

		if err := UnzipAnnotation(dataZipPath, unzipAnnotationPath); err != nil {
			returnChan <- kitendpoint.Response{IsLast: true, Data: nil, Err: err}
			return
		}

		buildFindOneResp = <-buildFindOne.Send(
			ctx,
			s.Conn,
			buildFindOne.RequestData{
				ProblemId: cvatTask.ProblemId,
				Status:    buildStatus.Tmp,
			},
		)
		tmpBuild = buildFindOneResp.Data.(buildFindOne.ResponseData)
		buildSplit = findBuildAssetSplit(tmpBuild, asset)
		cvatTask.Status = "pullReady"
		resp = <-cvatTaskUpdateOne.Send(context.TODO(), s.Conn, cvatTask)
		cvatTask = resp.Data.(cvatTaskUpdateOne.ResponseData)
		returnChan <- kitendpoint.Response{IsLast: true, Data: getCvatTaskAndAsset(cvatTask, asset, buildSplit), Err: nil}

	}()
	return returnChan
}

func makeTmpBuildPath(problemDir string) string {
	log.Printf("START: %s%s(%s)", fileLogPrefix, "makeTmpBuildPath", problemDir)
	path := fp.Join(problemDir, "/_builds/__tmp__")
	if err := os.MkdirAll(path, 0777); err != nil {
		log.Println("dump.makeTmpBuildPath.os.MkdirAll(path, 0777)", err)
	} else {
		log.Println("dump.makeTmpBuildPath.os.MkdirAll(path, 0777) created successfully")
	}
	log.Printf("FINISH: %s%s(%s) = %s", fileLogPrefix, "makeTmpBuildPath", problemDir, path)
	return path
}

func (s *basicCvatTaskService) getCvatTask(id primitive.ObjectID) t.CvatTask {
	log.Printf("START: %s%s(%s)", fileLogPrefix, "getCvatTask", id.Hex())
	cvatTaskFindOneResp := <-cvatTaskFindOne.Send(
		context.TODO(),
		s.Conn,
		cvatTaskFindOne.RequestData{
			Id: id,
		},
	)
	log.Printf("FINISH: %s%s(%s) = %v", fileLogPrefix, "getCvatTask1", id.Hex(), cvatTaskFindOneResp.Data.(cvatTaskFindOne.ResponseData))
	return cvatTaskFindOneResp.Data.(cvatTaskFindOne.ResponseData)
}

func (s *basicCvatTaskService) getAsset(id primitive.ObjectID) assetFindOne.ResponseData {
	log.Printf("START: %s%s(%s)", fileLogPrefix, "getAsset", id.Hex())
	assetFindOneResp := <-assetFindOne.Send(
		context.TODO(),
		s.Conn,
		assetFindOne.RequestData{
			Id: id,
		},
	)
	log.Printf("FINISH: %s%s(%s) = %v", fileLogPrefix, "getAsset", id.Hex(), assetFindOneResp.Data.(assetFindOne.ResponseData))
	return assetFindOneResp.Data.(assetFindOne.ResponseData)
}

func (s *basicCvatTaskService) getProblem(id primitive.ObjectID) problemFindOne.ResponseData {
	log.Printf("START: %s%s(%s)", fileLogPrefix, "getProblem", id.Hex())
	datasetFindOneResp := <-problemFindOne.Send(
		context.TODO(),
		s.Conn,
		problemFindOne.RequestData{
			Id: id,
		},
	)
	log.Printf("FINISH: %s%s(%s) = %v", fileLogPrefix, "getProblem", id.Hex(), datasetFindOneResp.Data.(problemFindOne.ResponseData))
	return datasetFindOneResp.Data.(problemFindOne.ResponseData)
}

func isFolder(asset t.Asset) bool {
	return asset.Type == typeAsset.Folder
}

func (s *basicCvatTaskService) getChildrenCvatTasks(asset t.Asset, problemId primitive.ObjectID) []t.CvatTask {
	log.Printf("START: %s%s(asset: %v, problemId: %s)", fileLogPrefix, "getChildrenCvatTasks", asset, problemId.Hex())
	assetFindResp := <-assetFind.Send(
		context.TODO(),
		s.Conn,
		assetFind.RequestData{
			ParentFolder: strings.Join([]string{asset.ParentFolder, asset.Name}, "/"),
		},
	)
	assets := assetFindResp.Data.(assetFind.ResponseData).Items
	var assetIds []primitive.ObjectID
	for _, asset := range assets {
		assetIds = append(assetIds, asset.Id)
	}
	cvatTaskFindResp := <-cvatTaskFind.Send(
		context.TODO(),
		s.Conn,
		cvatTaskFind.RequestData{
			AssetIds:  assetIds,
			ProblemId: problemId,
		},
	)
	log.Printf("FINISH: %s%s(asset: %v, problemId: %s) = %v", fileLogPrefix, "getChildrenCvatTasks", asset, problemId.Hex(), cvatTaskFindResp.Data.(cvatTaskFind.ResponseData).Items)
	return cvatTaskFindResp.Data.(cvatTaskFind.ResponseData).Items
}

func isCvatDatasetDumped(asset t.Asset) bool {
	datasetDumpFolderPath := getCvatDataDumpPath(asset)
	if _, err := os.Stat(datasetDumpFolderPath); !os.IsNotExist(err) {
		return true
	}
	return false
}

func UnzipAnnotation(src, dst string) error {
	log.Printf("START: %s%s(src: %s, dst: %s)", fileLogPrefix, "UnzipAnnotation", src, dst)
	defer log.Printf("FINISH: %s%s(src: %s, dst: %s)", fileLogPrefix, "UnzipAnnotation", src, dst)
	r, err := zip.OpenReader(src)
	if err != nil {
		log.Println("dump.UnzipAnnotation.zip.OpenReader(src)", err)
		return err
	}
	defer r.Close()

	for _, f := range r.File {
		log.Println(f.Name)
		if !strings.HasSuffix(f.Name, "instances_default.json") {
			continue
		}
		outFile, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			log.Println("dump.UnzipAnnotation.os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())", err)
			return err
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		_, err = io.Copy(outFile, rc)
		outFile.Close()
		rc.Close()
		if err != nil {
			return err
		}
		break
	}
	return nil
}

func UnzipDataset(src, dst string) error {
	log.Printf("START: %s%s(src: %s, dst: %s)", fileLogPrefix, "UnzipDataset", src, dst)
	defer log.Printf("FINISH: %s%s(src: %s, dst: %s)", fileLogPrefix, "UnzipDataset", src, dst)
	r, err := zip.OpenReader(src)
	if err != nil {
		log.Println("dump.UnzipDataset.zip.OpenReader(src)", err)
		if err := os.RemoveAll(dst); err != nil {
			log.Println("dump.UnzipDataset.os.RemoveAll(dst)", err)
		}
		return err
	}
	defer r.Close()
	imageFolder := "images/"

	for _, f := range r.File {
		if !strings.HasPrefix(f.Name, imageFolder) {
			continue
		}
		relFilePath, err := fp.Rel(imageFolder, f.Name)
		if err != nil {
			log.Println("domains.cvat_task.pkg.service.dump.UnzipDataset.fp.Rel(imageFolder, f.Name)", err)
		}
		outFilePath := fp.Join(dst, relFilePath)
		if err := os.MkdirAll(fp.Dir(outFilePath), 0777); err != nil {
			log.Println("dump.UnzipDataset.os.MkdirAll(fp.Dir(outFilePath), 0777)", err)
		}
		outFile, err := os.OpenFile(outFilePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			return err
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		_, err = io.Copy(outFile, rc)
		outFile.Close()
		rc.Close()
		if err != nil {
			return err
		}
	}
	return nil

}

func mkTmpDir() string {
	log.Printf("START: %s%s()", fileLogPrefix, "mkTmpDir")
	defer log.Printf("FINISH: %s%s()", fileLogPrefix, "mkTmpDir")
	path := fmt.Sprintf("/tmp/%d", rand.Int())
	if err := os.MkdirAll(path, 0777); err != nil {
		log.Println("domains.cvat_task.pkg.service.dump.mkTmpDir.os.copyAnnotationsFromTmpToBuildFolder", err)
		info, err := os.Stat("/tmp")
		if err != nil {
			log.Println("os.Stat(\"/tmp\")", err)
		} else {

			log.Println("os.Stat(\"/tmp\")", info.Mode())
		}
	} else {
		log.Println(path, "was created successfully")
	}
	return path
}

func removeTmpDir(path string) {
	log.Printf("START: %s%s(path: %s)", fileLogPrefix, "removeTmpDir", path)
	defer log.Printf("FINISH: %s%s(path: %s)", fileLogPrefix, "removeTmpDir", path)
	if err := os.RemoveAll(path); err != nil {
		log.Println("domains.cvat_task.pkg.service.dump.removeTmpDir.os.RemoveAll", err)
	}
}

func mkUnzipDatasetDir(asset t.Asset) string {
	log.Printf("START: %s%s(asset: %v)", fileLogPrefix, "mkUnzipDatasetDir", asset)
	path := getCvatDataDumpPath(asset)

	if err := os.MkdirAll(path, 0777); err != nil {
		log.Println("domains.cvat_task.pkg.service.dump.mkUnzipDatasetDir.os.MkdirAll(path, 0777)", err)
	}
	path = fmt.Sprintf("%s/", path)
	log.Printf("FINISH: %s%s(asset: %v) = %s", fileLogPrefix, "mkUnzipDatasetDir", asset, path)
	return path
}

func getCvatDataDumpPath(asset t.Asset) string {
	log.Printf("START: %s%s(asset: %v)", fileLogPrefix, "getCvatDataDumpPath", asset)
	folderName := fmt.Sprintf("%s/%s", asset.ParentFolder, asset.Name)
	folderName = strings.ReplaceAll(folderName, "/", "_")
	folderName = strings.ReplaceAll(folderName, " ", "_")
	folderName = strings.ReplaceAll(folderName, ".", "_")
	for _, l := range folderName {
		if l != '_' {
			break
		}
		folderName = folderName[1:]
	}
	path := fmt.Sprintf("/images/%s", folderName)
	log.Printf("FINISH: %s%s(asset: %v) = %s", fileLogPrefix, "getCvatDataDumpPath", asset, path)
	return path
}

func (s *basicCvatTaskService) saveCvatDataPathToAsset(asset t.Asset, cvatDataPath string) t.Asset {
	log.Printf("START: %s%s(asset: %v, cvatDataPath: %s)", fileLogPrefix, "saveCvatDataPathToAsset", asset, cvatDataPath)
	assetUpdateUpsertResp := <-assetUpdateUpsert.Send(context.TODO(), s.Conn, assetUpdateUpsert.RequestData{
		ParentFolder: asset.ParentFolder,
		Name:         asset.Name,
		Type:         asset.Type,
		CvatDataPath: cvatDataPath,
	})
	log.Printf("FINISH: %s%s(asset: %v, cvatDataPath: %s) = %v", fileLogPrefix, "saveCvatDataPathToAsset", asset, cvatDataPath, assetUpdateUpsertResp.Data.(t.Asset))
	return assetUpdateUpsertResp.Data.(t.Asset)
}
