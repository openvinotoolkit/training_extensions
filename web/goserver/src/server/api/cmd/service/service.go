package service

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	fp "path/filepath"

	"github.com/gorilla/websocket"
	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	"server/api/pkg/service"
	n "server/common/names"
	t "server/db/pkg/types"
	modelUpdateFromLocal "server/domains/model/pkg/handler/update_from_local"
	problemUpdateFromLocal "server/domains/problem/pkg/handler/update_from_local"

	kitutils "server/kit/utils"
	"server/kit/utils/basic/arrays"
)

var upgrader = websocket.Upgrader{}
var servicesPubQueues map[string]*amqp.Queue

var (
	conn             *rabbitmq.Connection
	rabbitCloseError chan *amqp.Error
)

func Run(httpAddr, amqpUser, amqpPass, amqpAddr, oteProblemsPath string) {
	log.Println("API Started")
	amqpUrl := fmt.Sprintf("amqp://%s:%s@%s/", amqpUser, amqpPass, amqpAddr)
	conn, err := rabbitmq.Dial(amqpUrl)
	if err != nil {
		log.Panic(err)
	}
	defer conn.Close()
	ch, err := conn.Channel()
	if err != nil {
		log.Panic(err)
	}
	defer ch.Close()

	servicesQueuesNames := []string{n.QAsset, n.QDatabase, n.QProblem, n.QTrainModel, n.QModel, n.QBuild}
	servicesPubQueues = kitutils.AmqpServicesQueuesDelare(conn, servicesQueuesNames)
	go reviseData(conn, oteProblemsPath)
	upgrader.CheckOrigin = func(r *http.Request) bool {
		return true
	}
	wsHandler := makeWsHandler(conn)
	http.HandleFunc("/api/ws", wsHandler)
	log.Fatal(http.ListenAndServe(httpAddr, nil))
	log.Println("THE END")
}

func makeWsHandler(conn *rabbitmq.Connection) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		log.Println("open WS")
		wsConn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			fmt.Println("WS Upgrage", err)
		}

		wsResponse := make(chan service.WSResponse)

		proxy := service.BasicProxy{
			wsConn,
			wsResponse,
			conn,
			servicesPubQueues,
			// serviceSubQueue,
		}
		ctx := r.Context()
		go proxy.WSRead(ctx)
		go proxy.WSWrite(ctx)
		<-ctx.Done()
		log.Println("Close ws")
	}
}

func reviseData(conn *rabbitmq.Connection, path string) {
	searchProblems(conn, path)
	searchModels(conn, path)
}

func searchProblems(conn *rabbitmq.Connection, root string) {
	err := fp.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.Println("api.cmd.servive.service.searchProblems.fp.Walk", err)
			return err
		}
		allowedProblemsFileNames := []string{"problems.yaml", "problems.yml"}
		if !arrays.ContainsString(allowedProblemsFileNames, info.Name()) {
			return nil
		}
		updateProblem(conn, path)
		return nil
	})
	if err != nil {
		log.Println("api.cmd.servive.service.searchProblems", err)
	}
}

func updateProblem(conn *rabbitmq.Connection, path string) []t.Problem {
	problemRes := <-problemUpdateFromLocal.Send(
		context.TODO(),
		conn,
		problemUpdateFromLocal.RequestData{
			Path: path,
		},
	)
	problems := problemRes.Data.(problemUpdateFromLocal.ResponseData)
	return problems
}

func searchModels(conn *rabbitmq.Connection, root string) {
	err := fp.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.Println("api.cmd.servive.service.searchModels.fp.Walk", err)
			return err
		}
		allowedModelsTemplateNames := []string{"template.yaml", "template.yml"}
		if !arrays.ContainsString(allowedModelsTemplateNames, info.Name()) {
			return nil
		}
		updateModel(conn, path)
		return nil
	})
	if err != nil {
		log.Println("api.cmd.servive.service.searchModels", err)
	}

}

func updateModel(conn *rabbitmq.Connection, path string) t.Model {
	modelRes := <-modelUpdateFromLocal.Send(
		context.TODO(),
		conn,
		modelUpdateFromLocal.RequestData{
			Path: path,
		},
	)
	model := modelRes.Data.(modelUpdateFromLocal.ResponseData)
	return model
}
