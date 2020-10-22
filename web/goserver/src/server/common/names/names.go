package names

// Websocket request/response event
// Communication between Client <-> API Service <-> Other services
const (
	EAssetDumpAnnotation = "ASSET_DUMP_ANNOTATION"
	EAssetFindInFolder   = "ASSET_FIND_IN_FOLDER"
	EAssetSetupToCvat    = "ASSET_SETUP_TO_CVAT"

	EBuildCreate           = "BUILD_CREATE"
	EBuildList             = "BUILD_LIST"
	EBuildUpdateAssetState = "BUILD_UPDATE_ASSET_STATE"

	EModelDelete   = "MODEL_DELETE"
	EModelEvaluate = "MODEL_EVALUATE"
	EModelFineTune = "MODEL_FINE_TUNE"
	EModelList     = "MODEL_LIST"

	EProblemCreate  = "PROBLEM_CREATE"
	EProblemDelete  = "PROBLEM_DELETE"
	EProblemDetails = "PROBLEM_DETAILS"
	EProblemList    = "PROBLEM_LIST"

	EUnsubscribe = "UNSUBSCRIBE"
)

// Amqp Queues names
const (
	QAsset           = "ASSET"
	QBuild           = "BUILD"
	QCvatTask        = "CVAT_TASK"
	QDatabase        = "DB"
	QDatabaseWatcher = "DB_WATCHER"
	QProblem         = "PROBLEM"
	QModel           = "MODEL"
	QTrainModel      = "TRAIN_MODEL"
)

// Mongodb collections names
const (
	CAsset    = "asset"
	CBuild    = "build"
	CCvatTask = "cvatTask"
	CProblem  = "problem"
	CModel    = "model"
)

// AMQP requests events
// Communication between services
// Asset service <-> DB service
const (
	RBuildCreateEmpty = "BUILD_CREATE_EMPTY"
	RBuildUpdateTmps  = "BUILD_UPDATE_TMPS"

	RDBAssetFindOne      = "DB_ASSET_FIND_ONE"
	RDBAssetFind         = "DB_ASSET_FIND"
	RDBAssetUpdateUpsert = "DB_ASSET_UPDATE_UPSERT"

	RDBBuildFind      = "DB_BUILD_FIND"
	RDBBuildFindOne   = "DB_BUILD_FIND_ONE"
	RDBBuildInsertOne = "DB_BUILD_INSERT_ONE"
	RDBBuildUpdateOne = "DB_BUILD_UPDATE_ONE"

	RDBCvatTaskFind      = "DB_CVAT_TASK_FIND"
	RDBCvatTaskFindOne   = "DB_CVAT_TASK_FIND_ONE"
	RDBCvatTaskInsertOne = "DB_CVAT_TASK_INSERT_ONE"
	RDBCvatTaskUpdateOne = "DB_CVAT_TASK_UPDATE_ONE"

	RDBProblemDelete       = "DB_PROBLEM_DELETE"
	RDBProblemFind         = "DB_PROBLEM_FIND"
	RDBProblemFindOne      = "DB_PROBLEM_FIND_ONE"
	RDBProblemUpdateUpsert = "DB_PROBLEM_UPDATE_UPSERT"

	RDBModelDelete       = "DB_MODEL_DELETE"
	RDBModelFind         = "DB_MODEL_FIND"
	RDBModelFindOne      = "DB_MODEL_FIND_ONE"
	RDBModelInsertOne    = "DB_MODEL_INSERT_ONE"
	RDBModelUpdateOne    = "DB_MODEL_UPDATE_ONE"
	RDBModelUpdateUpsert = "DB_MODEL_UPDATE_UPSERT"

	RModelCreateFromGeneric = "MODEL_CREATE_FROM_GENERIC"
	RModelUpdateFromLocal   = "MODEL_UPDATE_FROM_LOCAL"

	RProblemUpdateFromLocal = "PROBLEM_UPDATE_FROM_LOCAL"

	RTrainModelRunCommands  = "TRAIN_MODEL_RUN_COMMANDS"
	RTrainModelGetGpuAmount = "TRAIN_MODEL_GET_GPU_AMOUNT"
)

func GetEvents() map[string]string {
	return map[string]string{
		EAssetDumpAnnotation:   QCvatTask,
		EAssetFindInFolder:     QCvatTask,
		EAssetSetupToCvat:      QCvatTask,
		EBuildCreate:           QBuild,
		EBuildList:             QBuild,
		EBuildUpdateAssetState: QBuild,
		EModelDelete:           QModel,
		EModelEvaluate:         QModel,
		EModelList:             QModel,
		EModelFineTune:         QModel,
		EProblemCreate:         QProblem,
		EProblemDelete:         QProblem,
		EProblemDetails:        QProblem,
		EProblemList:           QProblem,
	}
}
