package background

import (
	"context"
	"fmt"
	"log"
	"os"
	fp "path/filepath"
	"regexp"
	"time"

	"go.mongodb.org/mongo-driver/bson/primitive"

	"github.com/radovskyb/watcher"

	assetFindOne "server/db/pkg/handler/asset/find_one"
	assetUpdateUpsert "server/db/pkg/handler/asset/update_upsert"
	typeAsset "server/db/pkg/types/type/asset"
	buildUpdateTmps "server/domains/build/pkg/handler/update_tmps"
)

func (b *basicAssetBackground) UpdateFromDisk(_ context.Context, root string, timeout time.Duration) {
	w := watcher.New()
	w.FilterOps(watcher.Rename, watcher.Move, watcher.Create, watcher.Write)
	go b.handleChanges(w, root)
	if err := w.AddRecursive(root); err != nil {
		log.Panicln("update_from_disk.UpdateFromDisk.AddRecursive(root)", err)
	}
	r := regexp.MustCompile(`^\w+(\.(?:(mp4|zip)))?$`)
	w.AddFilterHook(watcher.RegexFilterHook(r, false))
	go triggerInitEvent(w)
	if err := w.Start(timeout); err != nil {
		log.Panicln("update_from_disk.UpdateFromDisk.Start", err)
	}

}

func (b *basicAssetBackground) handleChanges(w *watcher.Watcher, root string) {
	for {
		select {
		case event := <-w.Event:
			log.Println("update_from_disk.handleChanges, event = ", event)
			b.updateAssetsFromDisk(root)
			<-buildUpdateTmps.Send(context.TODO(), b.Conn, buildUpdateTmps.RequestData{})
		case err := <-w.Error:
			log.Panicln("update_from_disk.handleChanges", err)
		case <-w.Closed:
			return
		}
	}
}

func (b *basicAssetBackground) updateAssetsFromDisk(root string) {

	err := fp.Walk(root, func(path string, info os.FileInfo, err error) error {
		var returnVal error
		var parentFolder string
		var name string
		var aType string
		if err != nil {
			log.Printf("prevent panic by handling failure accessing a path %q: %v\n", path, err)
			return err
		} else if path == root {
			log.Println("skip root folder", path)
			return nil
		}

		path, err = fp.Rel(root, path)
		if err != nil {
			log.Println("pgk.backgroun.update_from_disk.updateAssetsFromDisk.fp.Rel(root, path)")
		}

		if info.IsDir() {
			returnVal = nil
			parentFolder = relativeDir(path)
			name = info.Name()
			aType = typeAsset.Folder
		} else if fp.Ext(info.Name()) == ".jpg" || fp.Ext(info.Name()) == ".jpeg" {
			// save parent dir with type ImgDir
			returnVal = fp.SkipDir
			path = fp.Dir(path)
			parentFolder = relativeDir(path)
			name = fp.Base(path)
			aType = typeAsset.ImageFolder
		} else if fp.Ext(info.Name()) == ".zip" {
			returnVal = nil
			parentFolder = relativeDir(path)
			name = info.Name()
			aType = typeAsset.Archive
		} else if fp.Ext(info.Name()) == ".mp4" {
			returnVal = nil
			parentFolder = relativeDir(path)
			name = info.Name()
			aType = typeAsset.Video
		} else {
			return nil
		}
		assetResp := <-assetFindOne.Send(
			context.TODO(),
			b.Conn,
			assetFindOne.RequestData{
				ParentFolder: parentFolder,
				Name:         name,
			},
		)
		asset := assetResp.Data.(assetFindOne.ResponseData)
		if primitive.ObjectID.IsZero(asset.Id) || (asset.Type == typeAsset.Folder && aType == typeAsset.ImageFolder) {
			<-assetUpdateUpsert.Send(
				context.TODO(),
				b.Conn,
				assetUpdateUpsert.RequestData{
					ParentFolder: parentFolder,
					Name:         name,
					Type:         aType,
				},
			)
		}

		return returnVal
	})
	if err != nil {
		log.Println("server.domains.asset.pkg.background.update_from_disk.UpdateFromDisk.fp.Walk")
	}
}

func relativeDir(path string) string {
	parentFolder := fp.Dir(path)
	if parentFolder != "." {
		parentFolder = fmt.Sprintf("./%s", parentFolder)
	}
	return parentFolder
}

func triggerInitEvent(w *watcher.Watcher) {
	w.Wait()
	w.TriggerEvent(watcher.Write, nil)
}
